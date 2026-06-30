import io

import joblib
import lasio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Facies Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1923; }
    [data-testid="stSidebar"] * { color: #c9d8e8 !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #5ab4d6 !important; }
    h1 { color:white !important; letter-spacing: -0.5px; }
    [data-testid="metric-container"] {
        background: #f0f6fb;
        border: 1px solid #c5dde8;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: white;
        border-left: 4px solid #5ab4d6;
        padding-left: 10px;
        margin: 1.5rem 0 0.75rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Constants

FACIES_INFO = {
    1: {"name": "Nonmarine Sandstone",        "color": "#F4D03F"},
    2: {"name": "Nonmarine Coarse Silt",      "color": "#F5B041"},
    3: {"name": "Nonmarine Fine Silt",        "color": "#DC7633"},
    4: {"name": "Marine Siltstone",           "color": "#A11D33"},
    5: {"name": "Mudstone",                   "color": "#1B4F72"},
    6: {"name": "Wackestone",                 "color": "#2E4053"},
    7: {"name": "Dolomite",                   "color": "#7D6608"},
    8: {"name": "Packstone-Grainstone",       "color": "#117A65"},
    9: {"name": "Phylloid Algal Bafflestone", "color": "#145A32"},
}

FACIES_COLORS = [FACIES_INFO[i]["color"] for i in range(1, 10)]
CMAP_FACIES   = mcolors.ListedColormap(FACIES_COLORS, "indexed")

CURVE_ALIASES = {
    "GR": [
        "GR", "GGCE", "GR_ED", "GAM", "CGR", "SGR", "GRD", "GR_S", 
        "ECGR", "NGAM", "HGR", "EDTC_GR", "GRS", "GRMAIN", "GGR", "GAMMA",
        "NGR", "RGR", "GR_NNC", "GMA"
    ],
    "ILD_log10": [
        "ILD_LOG10", "RTAO", "ILD", "LL3", "RT", "AHT90", "AT90", "RILD", 
        "RLA5", "RD", "RDEEP", "RESDEEP", "HILD", "RT90", "M2R9", "HDIL",
        "LLD", "AHTD", "ATD", "M2R6", "RLA4", "M2R3", "RES", "RESD"
    ],
    "DeltaPHI": [
        "DELTAPHI", "DPHI", "DPOR", "DEPT_PHI", "DPHI_NPHI", "DPH", 
        "DPHZ", "POR_DENS", "DPOR_SAN", "DPOR_LIM", "DPHZ_L", "DPHI_LS", 
        "DPHI_SS", "DPHI_DOL", "DPH_LS", "DPH_SS", "DPOR_LS"
    ],
    "PHIND": [
        "PHIND", "XPOR", "NPHI", "PHIN", "NPHI_HL", "POROSITY", "NPOR", 
        "CNC", "CNLS", "HNPHI", "NPOR_SAN", "NPOR_LIM", "PHIN_L", "XPOR_LS",
        "NPHI_LS", "NPHI_SS", "NPHI_DOL", "TNPH", "NPHZ", "NPHZ_L"
    ],
    "PE": [
        "PE", "PDPE", "PEF", "DEN_COR", "PEFZ", "PECO", "PFE", 
        "PEF_SLB", "PEFZ_EDTC", "PDPE_L", "PE_B", "PEFZ_H", "EDTC_PEFZ"
    ]
}
REQUIRED_CURVES = list(CURVE_ALIASES.keys())

FEATURES_ORDERED = [
    "GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS",
    "GR_PHIND_ratio",
    "GR_roll_mean",        "GR_roll_std",
    "ILD_log10_roll_mean", "ILD_log10_roll_std",
    "DeltaPHI_roll_mean",  "DeltaPHI_roll_std",
    "PHIND_roll_mean",     "PHIND_roll_std",
    "PE_roll_mean",        "PE_roll_std",
]


# Model loading  (cached — runs once per session)

@st.cache_resource
def load_models():
    scaler   = joblib.load("scaler.joblib")
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

# Helper functions

def create_live_sequences(scaled_data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Pad a 2-D feature matrix and roll it into 3-D windows for CNN inference."""
    half_w = window_size // 2
    padded  = np.pad(scaled_data, ((half_w, half_w), (0, 0)), mode="edge")
    return np.array([padded[i : i + window_size] for i in range(len(scaled_data))])


def auto_map_curves(df: pd.DataFrame) -> dict:
    """Return {canonical_curve: detected_column_name | None} for every required curve."""
    mapping = {}
    for curve, aliases in CURVE_ALIASES.items():
        upper = [a.upper() for a in aliases]
        mapping[curve] = next((c for c in df.columns if c.upper() in upper), None)
    return mapping


def preprocess(df_las: pd.DataFrame, mapped_columns: dict) -> pd.DataFrame:
    """Build the model-ready feature frame from a raw LAS dataframe."""
    df  = pd.DataFrame()
    n   = len(df_las)
    df["Depth"] = df_las["Depth"].values

    for curve in REQUIRED_CURVES:
        col = mapped_columns.get(curve)
        df[curve] = (
            pd.to_numeric(df_las[col], errors="coerce").values
            if col and col in df_las.columns
            else np.full(n, np.nan)
        )

    # Auto-convert raw resistivity → log10 when values are clearly not already logged
    if df["ILD_log10"].median() > 5.0:
        df["ILD_log10"] = np.log10(df["ILD_log10"].clip(lower=0.001))

    # Geological metadata with safe defaults when absent from the file
    df["NM_M"] = (
        pd.to_numeric(df_las["NM_M"], errors="coerce").fillna(1).values
        if "NM_M" in df_las.columns
        else np.ones(n)
    )
    df["RELPOS"] = (
        pd.to_numeric(df_las["RELPOS"], errors="coerce").fillna(0.5).values
        if "RELPOS" in df_las.columns
        else np.full(n, 0.5)
    )

    # Median-fill any missing tool readings
    for col in REQUIRED_CURVES:
        med = df[col].median()
        df[col] = df[col].fillna(med if not np.isnan(med) else 0.0)

    # Engineered features
    df["GR_PHIND_ratio"] = df["GR"] / (df["PHIND"] + 0.001)

    for curve in REQUIRED_CURVES:
        roll = df[curve].rolling(window=3, min_periods=1)
        df[f"{curve}_roll_mean"] = roll.mean()
        df[f"{curve}_roll_std"]  = roll.std().fillna(0)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def facies_label(i: int) -> str:
    return f"{i} - {FACIES_INFO[i]['name']}" if i in FACIES_INFO else f"Class {i}"


def facies_color(i: int) -> str:
    return FACIES_INFO[i]["color"] if i in FACIES_INFO else "#aaaaaa"


def run_inference(model_name: str, X_scaled: np.ndarray) -> np.ndarray:
    """Route scaled features to the chosen model. Always returns 1-9 integer labels."""
    if model_name == "Random Forest":
        return rf_model.predict(X_scaled)

    if model_name == "XGBoost":
        if xgb_model is not None:
            # XGBoost was trained on 0-indexed labels (0-8); shift back to 1-9
            return xgb_model.predict(X_scaled) + 1
        st.sidebar.warning("XGBoost model not found — falling back to Random Forest.")
        return rf_model.predict(X_scaled)

    if model_name == "1D-CNN":
        if cnn_model is not None:
            X_seq = create_live_sequences(X_scaled, window_size=5)
            probs = cnn_model.predict(X_seq)
            # CNN was trained on 0-indexed labels; shift back to 1-9
            return np.argmax(probs, axis=1) + 1
        st.sidebar.warning("1D-CNN model not found — falling back to Random Forest.")
        return rf_model.predict(X_scaled)

    # Fallback
    return rf_model.predict(X_scaled)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_facies_legend():
    cols = st.columns(3)
    for idx, (fid, info) in enumerate(FACIES_INFO.items()):
        with cols[idx % 3]:
            st.markdown(
                f'<span style="display:inline-block;width:13px;height:13px;'
                f'background:{info["color"]};border-radius:2px;margin-right:6px;'
                f'vertical-align:middle;"></span>'
                f'<span style="font-size:0.82rem;">{fid} - {info["name"]}</span>',
                unsafe_allow_html=True,
            )


def render_log_strip(df: pd.DataFrame, df_las: pd.DataFrame, show_advanced: bool):
    n_tracks  = 5 if show_advanced else 3
    fig_width = 14 if show_advanced else 10
    fig, axes = plt.subplots(1, n_tracks, figsize=(fig_width, 10), sharey=True)
    depth     = df["Depth"]
    axes[0].invert_yaxis()

    # Track 0 - Gamma Ray
    axes[0].plot(df["GR"], depth, color="#1a3a52", lw=0.9)
    axes[0].set_title("GR", fontsize=9, fontweight="bold")
    axes[0].set_xlabel("API", fontsize=8)
    axes[0].set_ylabel("Depth", fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.4)

    # Track 1 - Resistivity
    axes[1].plot(df["ILD_log10"], depth, color="#1565c0", lw=0.9)
    axes[1].set_title("ILD", fontsize=9, fontweight="bold")
    axes[1].set_xlabel("Log10 Ohm-m", fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    if show_advanced:
        # Track 2 - PHIND
        axes[2].plot(df["PHIND"], depth, color="#2e7d32", lw=0.9)
        axes[2].set_title("PHIND", fontsize=9, fontweight="bold")
        axes[2].set_xlabel("v/v", fontsize=8)
        axes[2].grid(True, linestyle=":", alpha=0.4)

        # Track 3 - NM_M and RELPOS overlay
        nm   = df_las["NM_M"].values   if "NM_M"   in df_las.columns else np.ones(len(df))
        rpos = df_las["RELPOS"].values  if "RELPOS" in df_las.columns else np.linspace(0, 1, len(df))
        # Guard length mismatches caused by depth filtering
        nm   = nm[:len(depth)]   if len(nm)   > len(depth) else np.resize(nm,   len(depth))
        rpos = rpos[:len(depth)] if len(rpos) > len(depth) else np.resize(rpos, len(depth))
        axes[3].plot(nm,   depth, color="#6a1b9a", lw=1.2, label="NM_M")
        axes[3].plot(rpos, depth, color="#e65100", lw=0.8, ls="--", label="RELPOS")
        axes[3].set_title("Env / Pos", fontsize=9, fontweight="bold")
        axes[3].set_xlabel("Code / Index", fontsize=8)
        axes[3].legend(fontsize=7, loc="lower right")
        axes[3].grid(True, linestyle=":", alpha=0.4)

        facies_ax = axes[4]
    else:
        facies_ax = axes[2]

    # Predicted Facies strip
    pred  = df["Predicted_Facies"].values
    strip = np.repeat(pred, 100).reshape(-1, 100)
    facies_ax.imshow(
        strip, cmap=CMAP_FACIES, aspect="auto",
        extent=[0, 1, float(depth.max()), float(depth.min())],
        vmin=1, vmax=9,
    )
    facies_ax.set_title("Predicted Facies", fontsize=9, fontweight="bold")
    facies_ax.set_xticks([])

    plt.tight_layout(w_pad=0.3)
    st.pyplot(fig)
    plt.close(fig)

# Sidebar

with st.sidebar:
    st.markdown("## Controls")
    st.markdown("---")
    st.markdown("### Model")

    available_models = ["Random Forest"]
    if xgb_model is not None:
        available_models.append("XGBoost")
    if cnn_model is not None:
        available_models.append("1D-CNN")

    selected_model = st.selectbox("Interpretation engine", available_models)

    st.markdown("---")
    st.markdown("###  Display")
    show_advanced = st.checkbox("Show advanced tracks (PHIND, NM_M, RELPOS)", value=False)

    st.markdown("---")
    st.markdown("###  Model status")
    st.markdown(
        f"{'✅' if rf_model  is not None else '❌'} Random Forest\n\n"
        f"{'✅' if xgb_model is not None else '⬜'} XGBoost\n\n"
        f"{'✅' if cnn_model is not None else '⬜'} 1D-CNN"
    )


# Main area

st.title("Subsurface Facies Prediction")
st.caption(
    "Upload standard .las well log files to generate automated machine-learning "
    "lithofacies classifications."
)

uploaded_files = st.file_uploader(
    "Upload one or more well log files (.las)",
    type=["las"],
    accept_multiple_files=True,
    help="LAS 2.0 and LAS 3.0 formats are supported.",
)

if not uploaded_files:
    st.info(" Upload at least one .las file to begin.")
    st.stop()


# Per-well processing loop

for uploaded_file in uploaded_files:
    st.markdown("---")
    st.markdown(
        f"<div class='section-header'>Well: {uploaded_file.name}</div>",
        unsafe_allow_html=True,
    )

    # Parse LAS 
    try:
        raw    = uploaded_file.read()
        str_io = io.StringIO(raw.decode("utf-8", errors="ignore"))
        las    = lasio.read(str_io)
        df_las = las.df().reset_index()
        df_las.rename(columns={df_las.columns[0]: "Depth"}, inplace=True)
    except Exception as exc:
        st.error(f"Could not parse {uploaded_file.name}: {exc}")
        continue

    # Well header metadata
    with st.expander(" Well header metadata"):
        header_rows = [
            {"Mnemonic": item.mnemonic, "Unit": item.unit,
             "Value": item.value, "Description": item.descr}
            for item in las.well
        ]
        if header_rows:
            st.dataframe(pd.DataFrame(header_rows), use_container_width=True, hide_index=True)
        else:
            st.write("No header metadata found in this file.")

    # Curve mapping 
    auto_mapping   = auto_map_curves(df_las)
    mapped_columns = dict(auto_mapping)

    missing_curves = [c for c, col in auto_mapping.items() if col is None]
    if missing_curves:
        st.warning(
            f"⚠️ Could not auto-detect: {', '.join(missing_curves)}. "
            "Set them manually in the override panel below."
        )
    else:
        st.success(
            " Auto-mapped: " +
            ", ".join(f"**{c}** → `{col}`" for c, col in auto_mapping.items())
        )

    with st.expander("🔧 Curve mapping overrides"):
        cols = st.columns(len(REQUIRED_CURVES))
        for idx, curve in enumerate(REQUIRED_CURVES):
            with cols[idx]:
                default_col = mapped_columns.get(curve) or df_las.columns[0]
                default_idx = (
                    list(df_las.columns).index(default_col)
                    if default_col in df_las.columns else 0
                )
                mapped_columns[curve] = st.selectbox(
                    curve, df_las.columns, index=default_idx,
                    key=f"{uploaded_file.name}_{curve}",
                )

    # Warn if a curve ended up pointing at the depth column (i.e. truly missing)
    for curve, col in mapped_columns.items():
        if col in ("Depth", "DEPT"):
            st.sidebar.error(
                f"{uploaded_file.name}: no {curve} log found. "
                "Model accuracy will be reduced."
            )

    # Preprocessing 
    df_proc = preprocess(df_las, mapped_columns)

    # Depth range filter
    depth_min = float(df_proc["Depth"].min())
    depth_max = float(df_proc["Depth"].max())

    with st.expander("📏 Filter depth range"):
        d_lo, d_hi = st.slider(
            "Depth interval",
            min_value=depth_min, max_value=depth_max,
            value=(depth_min, depth_max),
            step=0.5,
            key=f"{uploaded_file.name}_depth",
        )

    mask       = (df_proc["Depth"] >= d_lo) & (df_proc["Depth"] <= d_hi)
    df_sub     = df_proc[mask].copy().reset_index(drop=True)
    df_las_sub = df_las[mask].reset_index(drop=True)

    if df_sub.empty:
        st.warning("No data in the selected depth range.")
        continue

    # Inference
    X_raw    = df_sub[FEATURES_ORDERED]
    X_scaled = scaler.transform(X_raw)
    df_sub["Predicted_Facies"] = run_inference(selected_model, X_scaled).astype(int)

    # Accuracy scorecard (only when ground-truth labels are present)
    if "FACIES" in df_las.columns:
        from sklearn.metrics import accuracy_score, classification_report

        true_labels = (
            pd.to_numeric(df_las_sub["FACIES"], errors="coerce")
            .fillna(-1).astype(int)
        )
        valid = true_labels != -1

        if valid.sum() > 0:
            acc = accuracy_score(true_labels[valid], df_sub["Predicted_Facies"][valid])
            st.markdown(
                "<div class='section-header'> Model scorecard</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Overall accuracy",    f"{acc:.1%}")
            c2.metric("Validated intervals", f"{valid.sum():,}")
            c3.metric("Depth range",         f"{d_lo:.0f} - {d_hi:.0f} ft")

            with st.expander(" Full classification report"):
                st.code(
                    classification_report(
                        true_labels[valid], df_sub["Predicted_Facies"][valid]
                    )
                )

    # Facies distribution bar chart
    st.markdown(
        "<div class='section-header'> Predicted facies distribution</div>",
        unsafe_allow_html=True,
    )
    counts = df_sub["Predicted_Facies"].value_counts().sort_index()
    bar_df  = pd.DataFrame({
        "Facies": [facies_label(i) for i in counts.index],
        "Count":  counts.values,
        "Color":  [facies_color(i) for i in counts.index],
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
    st.markdown(
        "<div class='section-header'> Log strip</div>",
        unsafe_allow_html=True,
    )
    render_log_strip(df_sub, df_las_sub, show_advanced)
    render_facies_legend()

    # GR vs Resistivity crossplot 
    st.markdown(
        "<div class='section-header'> GR vs Resistivity crossplot</div>",
        unsafe_allow_html=True,
    )
    df_sub["Facies_Label"] = df_sub["Predicted_Facies"].map(facies_label)
    fig_cross = px.scatter(
        df_sub,
        x="GR", y="ILD_log10",
        color="Facies_Label",
        color_discrete_map={
            f"{i} - {v['name']}": v["color"] for i, v in FACIES_INFO.items()
        },
        hover_data={"Depth": True, "PE": True, "PHIND": True, "Facies_Label": False},
        labels={
            "GR":          "Gamma Ray (API)",
            "ILD_log10":   "Resistivity (Log10 Ohm-m)",
            "Facies_Label": "Facies",
        },
        template="plotly_white",
        opacity=0.75,
    )
    fig_cross.update_traces(marker_size=4)
    fig_cross.update_layout(legend_title_text="Facies", margin=dict(t=30))
    st.plotly_chart(fig_cross, use_container_width=True)

    # CSV download
    st.markdown(
        "<div class='section-header'> Export results</div>",
        unsafe_allow_html=True,
    )
    stem = uploaded_file.name.rsplit(".", 1)[0]
    st.download_button(
        label=f"Download {stem}_facies.csv",
        data=convert_df_to_csv(df_sub),
        file_name=f"{stem}_facies.csv",
        mime="text/csv",
        key=f"dl_{uploaded_file.name}",
    )

st.markdown("---")
st.caption("Facies Predictor  |  built with Streamlit")
