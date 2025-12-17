# ==========================================================
# City-wide Crime Hotspot Prediction — Full Streamlit App
# Cross-version portable (Python 3.9+), CPU-friendly defaults
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import os, gc, sys, platform, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from scipy.stats import skew, kurtosis

import streamlit as st
import plotly.express as px

# --- API functions (must be importable in your env) ---
from dataRequest.crimedbRequest import (
    fetch_all_provinces,
    fetch_policestation_per_provinces,
    fetch_all_stats_province_quarterly,
)

# --- Optional XGBoost: fallback to GBR if not available ---
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    HAS_XGB = False

# ---------------- App/Theming ----------------
st.set_page_config(page_title="Crime Hotspot Prediction", layout="wide", page_icon="📊")
sns.set_theme(style="whitegrid", context="talk")

# Force black text globally for Matplotlib/Seaborn
plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "text.color": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#c7c7c7",
    "legend.edgecolor": "black",
})

# Helper: style axes in black (Matplotlib/Seaborn/SHAP)
def style_axes_black(ax):
    if ax is None:
        return
    # Spines
    for spine in ax.spines.values():
        spine.set_color("black")
    # Ticks/labels
    ax.tick_params(colors="black")
    if ax.xaxis and ax.xaxis.label: ax.xaxis.label.set_color("black")
    if ax.yaxis and ax.yaxis.label: ax.yaxis.label.set_color("black")
    # Title
    t = ax.title
    try:
        t.set_color("black")
    except Exception:
        pass
    # Legend (if any)
    leg = ax.get_legend()
    if leg is not None:
        try:
            leg.get_frame().set_edgecolor("black")
            for txt in leg.get_texts():
                txt.set_color("black")
            leg.set_title(leg.get_title().get_text() if leg.get_title() else None)
            if leg.get_title() is not None:
                leg.get_title().set_color("black")
        except Exception:
            pass

# Helper: force Plotly fonts to black for titles/axes/ticks/legend
def plotly_black(fig):
    fig.update_layout(
        font=dict(color="black"),
        legend=dict(font=dict(color="black"), title_font=dict(color="black")),
        title=dict(font=dict(color="black")),
    )
    # Handle multiple axes (subplot/facet-safe)
    for k in fig.layout:
        if k.startswith("xaxis") or k.startswith("yaxis"):
            ax = getattr(fig.layout, k)
            if ax is not None:
                if getattr(ax, "title", None):
                    ax.title.font = dict(color="black")
                ax.tickfont = dict(color="black")
                ax.linecolor = "black"
                ax.gridcolor = "#c7c7c7"
    return fig

# CSS to make Streamlit text components (incl. dataframes) black
st.markdown("""
<style>
/* Text & headers */
html, body, [class*="css"], .stMarkdown, .stText, .stCaption, .stHeader, .stSubheader { color: black !important; }
/* Dataframe/table */
[data-testid="stTable"], [data-testid="stDataFrame"] * { color: black !important; }
/* Plotly tick text (extra safety) */
.plotly .xtick text, .plotly .ytick text, .plotly .legendtext, .plotly .g-xtitle, .plotly .g-ytitle { fill: black !important; }
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
YEAR_MIN, YEAR_MAX = 2000, 2100

# Canonical 8 crime categories (used for category-focused SHAP and plots)
CANON_CATEGORIES = [
    "Total Sexual Offences",
    "Contact crime (crime against the person)",
    "TRIO Crime",
    "Contact-related crime",
    "Property-related crime",
    "Other serious crime",
    "17 Community reported serious crime",
    "Crime detected as a result of police action",
]

# ==========================================================
# Utility + Helpers
# ==========================================================
def gpu_info():
    """Best-effort GPU awareness (informational only)."""
    gpu = "CPU"
    try:
        import numba.cuda as _cuda  # noqa
        if _cuda.is_available():
            gpu = "CUDA GPU (numba)"
    except Exception:
        pass
    try:
        import cupy  # noqa
        gpu = "GPU (CuPy available)"
    except Exception:
        pass
    return gpu

def normalize_station_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure PoliceStationCode + StationName exist."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["PoliceStationCode", "StationName"])
    out = df.copy()
    # try common variants
    cols_lower = {c.lower(): c for c in out.columns}
    if "policestationcode" not in cols_lower:
        if "stationcode" in cols_lower:
            out.rename(columns={cols_lower["stationcode"]: "PoliceStationCode"}, inplace=True)
        elif "code" in cols_lower:
            out.rename(columns={cols_lower["code"]: "PoliceStationCode"}, inplace=True)
        else:
            out["PoliceStationCode"] = "UNKNOWN"
    if "stationname" not in cols_lower:
        if "name" in cols_lower:
            out.rename(columns={cols_lower["name"]: "StationName"}, inplace=True)
        else:
            out["StationName"] = "UNKNOWN"
    out["PoliceStationCode"] = out["PoliceStationCode"].astype(str)
    out["StationName"] = out["StationName"].astype(str)
    return out[["PoliceStationCode", "StationName"]]

def station_code_to_label(df_stations: pd.DataFrame, code: str) -> str:
    """Return 'CODE - Name' label if found, else 'CODE'."""
    if df_stations is None or df_stations.empty or "PoliceStationCode" not in df_stations.columns:
        return str(code)
    m = df_stations.loc[df_stations["PoliceStationCode"] == str(code)]
    if not m.empty and "StationName" in m.columns:
        return f"{code} - {m.iloc[0]['StationName']}"
    return str(code)

def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["PoliceStationCode", "StationName", "CrimeCategory", "QuarterofYear"])
    out = df.copy()
    req = [("PoliceStationCode", "UNKNOWN"),
           ("CrimeCategory", "UNKNOWN"),
           ("QuarterofYear", "1")]
    for c, default in req:
        if c not in out.columns:
            out[c] = default
    out["PoliceStationCode"] = out["PoliceStationCode"].astype(str)
    out["CrimeCategory"] = out["CrimeCategory"].astype(str)
    out["QuarterofYear"] = out["QuarterofYear"].astype(str)
    return out

def detect_year_cols(df: pd.DataFrame) -> list:
    years = []
    for c in df.columns:
        try:
            y = int(str(c))
            if YEAR_MIN <= y <= YEAR_MAX:
                years.append(c)
        except Exception:
            pass
    return sorted(years, key=lambda x: int(x))

def coerce_years_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in detect_year_cols(out):
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
    return out

def latest_year_in(df: pd.DataFrame) -> int:
    cols = detect_year_cols(df)
    return max([int(c) for c in cols]) if cols else 2024

def adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

# ==========================================================
# Feature Builder — Scenarios (always OHE CrimeCategory for SHAP y-axis)
# ==========================================================
def build_features_by_scenario(df_raw: pd.DataFrame, target_year: int, scenario: int):
    """
    X = all years strictly before target_year + Station/Quarter (scenario) + CrimeCategory (always OHE later)
    y = target_year
    Returns X, y, (num_cols, cat_cols)
    """
    df = ensure_core_columns(coerce_years_to_numeric(df_raw))
    year_cols = detect_year_cols(df)
    if str(target_year) not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), ([], [])
    hist_cols = [c for c in year_cols if int(c) < int(target_year)]
    if not hist_cols:
        return pd.DataFrame(), pd.Series(dtype=float), ([], [])

    y = df[str(target_year)].astype(float)
    X = df[hist_cols].copy()

    # attach Station/Quarter + CrimeCategory
    X["PoliceStationCode"] = df["PoliceStationCode"].astype(str)
    X["QuarterofYear"] = df["QuarterofYear"].astype(str)
    X["CrimeCategory"] = df["CrimeCategory"].astype(str)

    num_cols = [c for c in X.columns if c in hist_cols]  # numeric years
    cat_cols = ["CrimeCategory"]  # always OHE for categories (so SHAP shows categories)

    # scenario encoding for Station + Quarter
    def labelify(col):
        nonlocal X, num_cols, cat_cols
        X[col] = LabelEncoder().fit_transform(X[col].fillna("Unknown"))
        if col not in num_cols: num_cols.append(col)
        if col in cat_cols: cat_cols.remove(col)

    def onehotify(col):
        nonlocal cat_cols
        if col not in cat_cols:
            cat_cols.append(col)

    if scenario == 1:          # Label + Label
        labelify("PoliceStationCode"); labelify("QuarterofYear")
    elif scenario == 2:        # OHE + OHE
        onehotify("PoliceStationCode"); onehotify("QuarterofYear")
    elif scenario == 3:        # Label + OHE
        labelify("PoliceStationCode"); onehotify("QuarterofYear")
    elif scenario == 4:        # OHE + Label
        onehotify("PoliceStationCode"); labelify("QuarterofYear")
    else:                      # default OHE + OHE
        onehotify("PoliceStationCode"); onehotify("QuarterofYear")

    return X, y, (num_cols, cat_cols)

def make_pipeline(model, num_cols, cat_cols):
    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),  # normalization is always ON
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("prep", pre), ("model", model)])

def model_dict():
    models = {
        "RFM": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "SVR": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
        "KNNR": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "MLPR": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGB"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",     # fast on CPU; uses GPU if configured
            random_state=RANDOM_STATE,
            eval_metric="rmse",
        )
    else:
        models["GBR"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
    return models

def param_grid():
    return {
        "RFM": {"model__n_estimators": [200], "model__max_depth": [None, 10]},
        "SVR": {"model__C": [1, 10]},
        "KNNR": {"model__n_neighbors": [5, 7, 9]},
        "MLPR": {"model__max_iter": [800]},
        "XGB": {"model__n_estimators": [300]},
        "GBR": {"model__n_estimators": [300]},
    }

# ==========================================================
# Caching Fetches
# ==========================================================
@st.cache_data(ttl=600, show_spinner=False)
def cached_provinces():
    try:
        df = pd.DataFrame(fetch_all_provinces())
        if "ProvinceName" in df: df["ProvinceName"] = df["ProvinceName"].astype(str)
        if "ProvinceCode" in df: df["ProvinceCode"] = df["ProvinceCode"].astype(str)
        return df
    except Exception as e:
        st.error(f"Failed to fetch provinces: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def cached_stations(province_code: str):
    try:
        return normalize_station_df(pd.DataFrame(fetch_policestation_per_provinces(province_code)))
    except Exception as e:
        st.error(f"Failed to fetch stations for {province_code}: {e}")
        return pd.DataFrame(columns=["PoliceStationCode", "StationName"])

@st.cache_data(ttl=600, show_spinner=False)
def cached_quarter(province_code: str, quarter: int):
    try:
        df = pd.DataFrame(fetch_all_stats_province_quarterly(province_code, quarter))
        return ensure_core_columns(coerce_years_to_numeric(df))
    except Exception as e:
        st.error(f"Failed to fetch Quarter {quarter}: {e}")
        return ensure_core_columns(pd.DataFrame())

# ==========================================================
# Sidebar — Settings
# ==========================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Runtime: {platform.system()} • Python {platform.python_version()} • Compute: {gpu_info()}")

    df_prov = cached_provinces()
    if df_prov.empty or not {"ProvinceName","ProvinceCode"}.issubset(df_prov.columns):
        st.error("Province list unavailable; defaulting to Western Cape.")
        province_name, province_code = "Western Cape", "ZA.WC"
    else:
        default_idx = 0
        if "ZA.WC" in df_prov["ProvinceCode"].values:
            default_idx = int(df_prov.index[df_prov["ProvinceCode"] == "ZA.WC"][0])
        province_name = st.selectbox("Province", df_prov["ProvinceName"], index=default_idx, key="sb_province")
        province_code = df_prov.loc[df_prov["ProvinceName"] == province_name, "ProvinceCode"].values[0]

    df_stations = cached_stations(province_code)
    if df_stations.empty:
        st.warning(f"No stations found for {province_name} ({province_code}); using UNKNOWN.")
        station_name = st.selectbox("Police Station", ["UNKNOWN"], index=0, key="sb_station")
        station_code = "UNKNOWN"
    else:
        labels = [f"{r.PoliceStationCode} - {r.StationName}" for _, r in df_stations.iterrows()]
        station_label = st.selectbox("Police Station", labels, index=0, key="sb_station")
        station_code = station_label.split(" - ")[0]
        station_name = station_label

st.session_state["selected_province_code"] = province_code
st.session_state["selected_province_name"] = province_name
st.session_state["selected_station_code"] = station_code
st.session_state["selected_station_label"] = station_name

# ==========================================================
# Load 4 quarters
# ==========================================================
with st.spinner(f"Loading Q1–Q4 for {province_name} ({province_code})…"):
    quarters = {q: cached_quarter(province_code, q) for q in [1,2,3,4]}
gc.collect()

# ==========================================================
# Tabs
# ==========================================================
tab_about, tab_quarters, tab_clean, tab_outliers, tab_train, tab_xai, tab_forecast, tab_compare = st.tabs([
    "ℹ️ About", "📊 Quarterly Data", "🧹 Cleaning", "🔥 Outliers", "🤖 Training", "🧠 SHAP & LIME", "🔮 Forecast", "🏷️ Compare Stations"
])

# ==========================================================
# TAB: About
# ==========================================================
with tab_about:
    st.title("🏙️ City-wide Crime Hotspot Prediction")
    st.markdown("""
- **Scope:** 2016–2024 train, 2025–2027 forecast  
- **Models:** RFM, SVR, KNNR, MLPR, XGB/GBR  
- **Splits:** 60% Train, 20% Val, 20% Test (fixed)  
- **Scenarios (Station × Quarter):** 1) Label/Label  2) OHE/OHE  3) Label/OHE  4) OHE/Label  
- **Encoding:** Always OHE **CrimeCategory** so SHAP shows categories on Y-axis  
- **Metrics:** MSE, RMSE, MAE, MAPE, R², AdjR², Accuracy%, CV%  
- **Interpretability:** SHAP (category-focused), LIME (sampled)  
- **Forecast:** Per category & combined, **every year labeled**  
    """)

# ==========================================================
# TAB: Quarterly Data
# ==========================================================
with tab_quarters:
    st.subheader(f"Quarterly Data — {province_name} ({province_code})")
    cols = st.columns(4)
    for i, q in enumerate([1,2,3,4]):
        with cols[i]:
            st.metric(f"Q{q} records", len(quarters[q]))
    t1, t2, t3, t4 = st.tabs(["Q1", "Q2", "Q3", "Q4"])

    def show_quarter(df, q):
        if df.empty:
            st.warning(f"No data for Q{q}")
            return
        st.dataframe(df, use_container_width=True)
        # Heatmap: CrimeCategory × Year
        df2 = coerce_years_to_numeric(df)
        years = detect_year_cols(df2)
        if "CrimeCategory" in df2.columns and years:
            long = df2.melt(id_vars=["CrimeCategory"], value_vars=years, var_name="Year", value_name="Value")
            agg = long.groupby(["CrimeCategory","Year"])["Value"].sum().reset_index()
            mat = agg.pivot(index="CrimeCategory", columns="Year", values="Value").fillna(0)
            fig = px.imshow(mat, aspect="auto", color_continuous_scale="RdBu",
                            title=f"Q{q}: CrimeCategory × Year — Raw")
            plotly_black(fig)
            st.plotly_chart(fig, use_container_width=True)

    with t1: show_quarter(quarters[1], 1)
    with t2: show_quarter(quarters[2], 2)
    with t3: show_quarter(quarters[3], 3)
    with t4: show_quarter(quarters[4], 4)

# ==========================================================
# TAB: Cleaning (ffill/bfill, detailed)
# ==========================================================
with tab_clean:
    st.subheader("Forward & Backward Fill — Detailed Missing Analysis")
    st.session_state.setdefault("clean_quarters", {})
    for q, df in quarters.items():
        if df.empty: continue
        st.markdown(f"### Quarter {q}")
        df2 = df.copy()
        years = detect_year_cols(df2)
        # Missing BEFORE
        missing_before = df2.isnull().sum().reset_index()
        missing_before.columns = ["Column", "MissingCount"]
        missing_before["MissingPercent"] = (missing_before["MissingCount"] / max(len(df2),1) * 100).round(2)

        # Fill across year axis (row-wise)
        if years:
            df2[years] = df2[years].ffill(axis=1).bfill(axis=1)
        else:
            df2 = df2.fillna(method="ffill").fillna(method="bfill")

        # Missing AFTER
        missing_after = df2.isnull().sum().reset_index()
        missing_after.columns = ["Column", "MissingCount"]
        missing_after["MissingPercent"] = (missing_after["MissingCount"] / max(len(df2),1) * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Missing — Before**")
            st.dataframe(missing_before, use_container_width=True)
        with c2:
            st.write("**Missing — After**")
            st.dataframe(missing_after, use_container_width=True)

        st.session_state["clean_quarters"][q] = ensure_core_columns(df2)
        st.markdown("---")
gc.collect()

# ==========================================================
# Outliers helpers
# ==========================================================
def detect_and_replace_outliers(df: pd.DataFrame, method="IQR", z_thresh=3.0, lower_pct=0.01, upper_pct=0.99):
    df2 = coerce_years_to_numeric(df)
    years = detect_year_cols(df2)
    if not years:
        return df2, pd.DataFrame(), (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    dfc = df2.copy()
    summary_rows = []
    for col in years:
        s = dfc[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        li, ui = q1 - 1.5*iqr, q3 + 1.5*iqr
        mean, std = s.mean(), s.std()

        if method == "IQR":
            lo, hi = li, ui
        elif method == "Z-Score":
            lo, hi = mean - z_thresh*std, mean + z_thresh*std
        elif method == "Percentile Winsorization":
            lo, hi = s.quantile(lower_pct), s.quantile(upper_pct)
        else:
            lo, hi = li, ui

        mask = (s < lo) | (s > hi)
        if mask.any():
            dfc[col] = s.clip(lo, hi)

        summary_rows.append({
            "Column": col,
            "LowerBound": round(lo, 2),
            "UpperBound": round(hi, 2),
            "OutlierCount": int(mask.sum()),
            "Method": method,
        })
    summary_df = pd.DataFrame(summary_rows)

    def stats_block(df_):
        rows = []
        for col in years:
            series = df_[col].dropna().astype(float)
            vals = series.values
            rows.append({
                "Column": col,
                "Mean": series.mean(),
                "StdDev": series.std(),
                "Skew": skew(vals) if len(vals) > 2 else np.nan,
                "Kurtosis": kurtosis(vals) if len(vals) > 2 else np.nan,
            })
        return pd.DataFrame(rows)

    before = stats_block(df2)
    after = stats_block(dfc)
    delta = before.merge(after, on="Column", suffixes=("_Before", "_After"))
    for m in ["Mean", "StdDev", "Skew", "Kurtosis"]:
        delta[f"{m}_Change_%"] = (
            (delta[f"{m}_After"] - delta[f"{m}_Before"]) / delta[f"{m}_Before"].replace(0, np.nan)
        ) * 100
    delta = delta.round(3)

    return dfc, summary_df, (before, after, delta)

# ==========================================================
# TAB: Outliers — Heatmaps (before/after) + Boxplots (after)
# ==========================================================
with tab_outliers:
    st.subheader("Outliers — Cleaned Dataset (Heatmaps Before/After + Boxplots After)")
    st.session_state.setdefault("out_quarters", {})
    for q, df in st.session_state.get("clean_quarters", {}).items():
        if df.empty: continue
        st.markdown(f"### Quarter {q}")
        # Outlier replacement on cleaned data
        df_out, summary_df, (stats_before, stats_after, delta_df) = detect_and_replace_outliers(df, method="IQR")
        st.session_state["out_quarters"][q] = df_out

        # Heatmaps side-by-side
        dfB, dfA = df.copy(), df_out.copy()
        yearsB, yearsA = detect_year_cols(dfB), detect_year_cols(dfA)
        c1, c2 = st.columns(2)
        if "CrimeCategory" in dfB.columns and yearsB:
            with c1:
                matB = dfB.melt(id_vars=["CrimeCategory"], value_vars=yearsB, var_name="Year", value_name="Value")
                matB = matB.groupby(["CrimeCategory","Year"])["Value"].sum().reset_index()
                matB = matB.pivot(index="CrimeCategory", columns="Year", values="Value").fillna(0)
                figB = px.imshow(matB, aspect="auto", color_continuous_scale="RdBu",
                                 title=f"Q{q} — Before Outlier Replacement")
                plotly_black(figB)
                st.plotly_chart(figB, use_container_width=True)
        if "CrimeCategory" in dfA.columns and yearsA:
            with c2:
                matA = dfA.melt(id_vars=["CrimeCategory"], value_vars=yearsA, var_name="Year", value_name="Value")
                matA = matA.groupby(["CrimeCategory","Year"])["Value"].sum().reset_index()
                matA = matA.pivot(index="CrimeCategory", columns="Year", values="Value").fillna(0)
                figA = px.imshow(matA, aspect="auto", color_continuous_scale="RdBu",
                                 title=f"Q{q} — After Outlier Replacement")
                plotly_black(figA)
                st.plotly_chart(figA, use_container_width=True)

        # Tables
        st.write("**Outlier Summary**")
        st.dataframe(summary_df, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.write("**Stats — Before**")
            st.dataframe(stats_before, use_container_width=True)
        with c4:
            st.write("**Stats — After**")
            st.dataframe(stats_after, use_container_width=True)
        st.write("**Δ-Change (%)**")
        st.dataframe(delta_df, use_container_width=True)

        # Boxplots on AFTER dataset (cleaner view)
        num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.markdown("**Boxplots — After Outlier Replacement (numeric year columns)**")
            chunks = [num_cols[i:i+10] for i in range(0, len(num_cols), 10)]
            for idx, group in enumerate(chunks):
                fig, ax = plt.subplots(figsize=(min(12, 1.2*len(group)), 4))
                sns.boxplot(data=df_out[group], ax=ax, palette="Blues")
                ax.set_title(f"Q{q} — Boxplot group {idx+1}")
                ax.tick_params(axis='x', rotation=45)
                style_axes_black(ax)
                st.pyplot(fig)

        st.markdown("---")
gc.collect()

# ==========================================================
# TAB: TRAINING — 60/20/20 + Accuracy% + CV%
# ==========================================================
with tab_train:
    st.subheader("Model Training — Scenarios 1–4 (60/20/20) + Accuracy% & Cross-Validation%")

    quarters_for_model = st.session_state.get("out_quarters", st.session_state.get("clean_quarters", quarters))
    if st.button("🚀 Run Training (All Scenarios × 4 Quarters × 5 Models)", key="run_train_all"):
        all_metrics = []
        trained_models = {}

        for scenario in [1, 2, 3, 4]:
            st.markdown(f"### 🧩 Scenario {scenario}")
            scenario_rows = []

            for q in [1, 2, 3, 4]:
                dfq = quarters_for_model.get(q, pd.DataFrame())
                if dfq.empty:
                    st.warning(f"Q{q}: No data available.")
                    continue

                dfq = ensure_core_columns(coerce_years_to_numeric(dfq))
                ty = latest_year_in(dfq)
                X, y, (num_cols, cat_cols) = build_features_by_scenario(dfq, ty, scenario)
                if X.empty or y.empty:
                    st.warning(f"Q{q}: No features/target for year {ty}.")
                    continue

                # 60/20/20 Splits
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.4, random_state=RANDOM_STATE
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
                )

                # Small previews
                pv1, pv2, pv3 = st.columns(3)
                with pv1: st.caption("**Train**"); st.dataframe(pd.concat([X_train.head(5), y_train.head(5)], axis=1), use_container_width=True)
                with pv2: st.caption("**Val**");   st.dataframe(pd.concat([X_val.head(5),   y_val.head(5)],   axis=1), use_container_width=True)
                with pv3: st.caption("**Test**");  st.dataframe(pd.concat([X_test.head(5),  y_test.head(5)],  axis=1), use_container_width=True)

                pipe_models = model_dict()
                grids = param_grid()
                kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

                for name, mdl in pipe_models.items():
                    pipe = make_pipeline(mdl, num_cols, cat_cols)
                    grid = grids.get(name if name != "GBR" else "GBR", {})

                    with joblib.parallel_backend("threading"):
                        gs = GridSearchCV(pipe, grid, cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                                          scoring="r2", n_jobs=-1)
                        gs.fit(X_train, y_train)
                    best_est = gs.best_estimator_
                    trained_models[(scenario, q, name)] = best_est

                    # Train CV (R²) on training set
                    with joblib.parallel_backend("threading"):
                        cv_scores = cross_val_score(best_est, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)
                    cv_r2_mean = float(np.nanmean(cv_scores)) if len(cv_scores) else np.nan
                    cv_r2_std  = float(np.nanstd(cv_scores)) if len(cv_scores) else np.nan
                    cv_pct     = float(np.clip(cv_r2_mean * 100.0, 0.0, 100.0)) if not np.isnan(cv_r2_mean) else np.nan

                    def eval_split(split_name, Xs, ys):
                        yp = best_est.predict(Xs)
                        mse_  = mean_squared_error(ys, yp)
                        rmse_ = float(np.sqrt(mse_))
                        mae_  = mean_absolute_error(ys, yp)
                        r2_   = r2_score(ys, yp)
                        try:
                            k = best_est.named_steps["prep"].transform(X_train).shape[1]
                        except Exception:
                            k = len(num_cols) + len(cat_cols)
                        adj_  = adjusted_r2(r2_, len(ys), k)
                        mape_ = safe_mape(ys, yp)
                        acc_  = float(np.clip(100.0 - mape_, 0.0, 100.0)) if not np.isnan(mape_) else np.nan
                        return {
                            "Scenario": scenario, "Quarter": q, "Model": name, "Split": split_name,
                            "MSE": mse_, "RMSE": rmse_, "MAE": mae_, "MAPE": mape_, "R2": r2_, "AdjR2": adj_,
                            "Accuracy%": acc_, "CV_R2_mean": cv_r2_mean, "CV_R2_std": cv_r2_std, "CV%": cv_pct
                        }, yp

                    m_tr, _ = eval_split("Train", X_train, y_train)
                    m_va, yhat_val = eval_split("Val",   X_val,   y_val)
                    m_te, yhat_tst = eval_split("Test",  X_test,  y_test)
                    scenario_rows.extend([m_tr, m_va, m_te])

                    # ==== True vs Pred (Val & Test) with dark blues ====
                    g1, g2 = st.columns(2)
                    with g1:
                        fig, ax = plt.subplots(figsize=(5.2, 3.2))
                        ax.scatter(y_val, yhat_val, s=18, alpha=0.92, color="#0b5394")
                        lo, hi = min(y_val.min(), yhat_val.min()), max(y_val.max(), yhat_val.max())
                        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="#777777")
                        ax.set_title(f"{name} — Q{q} (Validation)")
                        ax.set_xlabel("True"); ax.set_ylabel("Pred")
                        ax.legend([f"R²={m_va['R2']:.3f}  RMSE={m_va['RMSE']:.3f}"],
                                  bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                        ax.grid(axis="both", linestyle=":", linewidth=0.6)
                        style_axes_black(ax)
                        st.pyplot(fig)
                    with g2:
                        fig, ax = plt.subplots(figsize=(5.2, 3.2))
                        ax.scatter(y_test, yhat_tst, s=18, alpha=0.92, color="#1f4e79")
                        lo, hi = min(y_test.min(), yhat_tst.min()), max(y_test.max(), yhat_tst.max())
                        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="#777777")
                        ax.set_title(f"{name} — Q{q} (Test)")
                        ax.set_xlabel("True"); ax.set_ylabel("Pred")
                        ax.legend([f"R²={m_te['R2']:.3f}  RMSE={m_te['RMSE']:.3f}"],
                                  bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                        ax.grid(axis="both", linestyle=":", linewidth=0.6)
                        style_axes_black(ax)
                        st.pyplot(fig)

            # ===== Per-scenario tables + FULL metric comparisons (Test) =====
            if scenario_rows:
                df_metrics = pd.DataFrame(scenario_rows)
                all_metrics.append(df_metrics)

                st.markdown("**Per-scenario Metrics (Train/Val/Test rows)**")
                st.dataframe(df_metrics.round(4), use_container_width=True)

                # Compact grids of Test metrics (bars)
                test_df = df_metrics[df_metrics["Split"] == "Test"].copy()

                left_col, right_col = st.columns(2)
                def metric_bar(ax, metric_name):
                    sns.barplot(data=test_df, x="Model", y=metric_name, ax=ax, palette="Blues")
                    ax.set_title(f"Scenario {scenario} — {metric_name} (Test)")
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", rotation=0)
                    ax.grid(axis="y", linestyle=":", linewidth=0.6)
                    style_axes_black(ax)

                with left_col:
                    fig, axs = plt.subplots(3, 1, figsize=(6.2, 7.8))
                    metric_bar(axs[0], "RMSE")
                    metric_bar(axs[1], "MSE")
                    metric_bar(axs[2], "MAE")
                    plt.tight_layout(); st.pyplot(fig)

                with right_col:
                    fig, axs = plt.subplots(3, 1, figsize=(6.2, 7.8))
                    metric_bar(axs[0], "MAPE")
                    metric_bar(axs[1], "R2")
                    metric_bar(axs[2], "AdjR2")
                    plt.tight_layout(); st.pyplot(fig)

                # Accuracy% & CV%
                st.markdown("**Accuracy% & Cross-Validation% (Test split / CV on train)**")
                acc_cv_cols = st.columns(2)

                with acc_cv_cols[0]:
                    fig, ax = plt.subplots(figsize=(6.6, 3.3))
                    sns.barplot(data=test_df, x="Model", y="Accuracy%", hue="Quarter", ax=ax, palette="Blues")
                    ax.set_title(f"Scenario {scenario} — Accuracy% (from MAPE) — Test")
                    ax.set_xlabel(""); ax.set_ylim(0, 100)
                    ax.grid(axis="y", linestyle=":", linewidth=0.6)
                    style_axes_black(ax)
                    # Legend in black
                    leg = ax.legend(title="Quarter", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                    if leg:
                        for txt in leg.get_texts(): txt.set_color("black")
                        if leg.get_title(): leg.get_title().set_color("black")
                    st.pyplot(fig)

                with acc_cv_cols[1]:
                    cv_view = test_df.groupby(["Model"], as_index=False)[["CV%", "CV_R2_mean", "CV_R2_std"]].mean()
                    fig, ax = plt.subplots(figsize=(6.6, 3.3))
                    sns.barplot(data=cv_view, x="Model", y="CV%", ax=ax, palette="Blues")
                    for i, row in cv_view.iterrows():
                        ax.text(i, row["CV%"] + 1.0, f"±{row['CV_R2_std']*100:.1f}", ha="center", va="bottom", fontsize=9, color="black")
                    ax.set_title(f"Scenario {scenario} — Cross-Validation% (R² mean × 100) — Train CV")
                    ax.set_xlabel(""); ax.set_ylim(0, 120)
                    ax.grid(axis="y", linestyle=":", linewidth=0.6)
                    style_axes_black(ax)
                    st.pyplot(fig)

                # Multi-metric line (Test) overview
                df_long = test_df.melt(
                    id_vars=["Model","Quarter"],
                    value_vars=["MSE", "RMSE", "MAE", "MAPE", "R2", "AdjR2", "Accuracy%"],
                    var_name="Metric", value_name="Value"
                )
                fig, ax = plt.subplots(figsize=(8.0, 3.6))
                sns.lineplot(data=df_long, x="Metric", y="Value", hue="Model",
                             marker="o", linewidth=1.2, ax=ax)
                ax.set_title(f"Scenario {scenario} — Metrics (Test)")
                ax.grid(axis="y", linestyle=":", linewidth=0.6)
                style_axes_black(ax)
                # Legend in black
                leg = ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
                if leg:
                    for txt in leg.get_texts(): txt.set_color("black")
                    if leg.get_title(): leg.get_title().set_color("black")
                st.pyplot(fig)

        # ------- Combine All Scenarios/Quarters -------
        if all_metrics:
            combined = pd.concat(all_metrics, ignore_index=True)
            st.session_state["combined_metrics"] = combined
            st.session_state["trained_models"] = trained_models

            st.header("📊 Combined Metrics (All Scenarios & Quarters)")
            st.dataframe(combined.round(4), use_container_width=True)

            # Select quarter for candidate picking (keep real quarter, no -1)
            quarter_for_candidates = st.selectbox(
                "Choose quarter for top-2 candidate selection",
                options=[1,2,3,4],
                index=0,
                key="top2_quarter_pick"
            )
            comb_test_q = combined[(combined["Split"] == "Test") & (combined["Quarter"] == quarter_for_candidates)].copy()

            # Rank by RMSE asc, R2 desc and pick 2 distinct models
            ranked = comb_test_q.sort_values(["RMSE","R2"], ascending=[True, False]).reset_index(drop=True)
            top_list, seen = [], set()
            for _, row in ranked.iterrows():
                if row["Model"] not in seen:
                    top_list.append(row)
                    seen.add(row["Model"])
                if len(top_list) == 2:
                    break
            if not top_list:
                st.warning("No models found to select top-2.")
            else:
                top2 = pd.DataFrame(top_list)
                st.session_state["top2_overall"] = top2[["Scenario","Quarter","Model"]]
                st.success("Top-2 Candidates: " + ", ".join(top2["Model"].tolist()))
        else:
            st.warning("No metrics to combine. Check your data/API.")

# ==========================================================
# TAB: SHAP & LIME — Review both candidates, then Accept one
# ==========================================================
with tab_xai:
    st.subheader("Explainability: Review the Top-2 then Accept One (no retrain on forecast)")

    combined = st.session_state.get("combined_metrics", pd.DataFrame())
    trained = st.session_state.get("trained_models", {})
    quarters_for_model = st.session_state.get("out_quarters", st.session_state.get("clean_quarters", quarters))

    if combined.empty or "top2_overall" not in st.session_state:
        st.info("Run training first and pick the quarter for top-2 candidates.")
    else:
        cand_df = st.session_state["top2_overall"].reset_index(drop=True)
        st.write("**Candidates (from selected quarter in Training tab):**")
        st.dataframe(cand_df, use_container_width=True)

        # SHAP/LIME sample size (kept to avoid timeouts with KernelExplainer)
        st.caption("SHAP/LIME use a sample for safety (increase if your machine can handle it).")
        shap_sample = st.slider("Max rows for SHAP/LIME sample", 100, 2000, 600, 50, key="xai_sample_cap")

        explain_blocks = []
        for idx in range(min(2, len(cand_df))):
            cm = cand_df.iloc[idx]["Model"]
            cs = int(cand_df.iloc[idx]["Scenario"])
            cq = int(cand_df.iloc[idx]["Quarter"])

            est = trained.get((cs, cq, cm))
            if est is None:
                st.warning(f"{cm} (Scenario {cs}) not found; skipping.")
                continue

            dfq = quarters_for_model.get(cq, pd.DataFrame())
            if dfq.empty:
                st.warning(f"No data for Q{cq}.")
                continue
            ty = latest_year_in(dfq)
            X_raw, y_raw, (num_cols, cat_cols) = build_features_by_scenario(dfq, ty, cs)
            if X_raw.empty:
                st.warning(f"No valid features for {cm}.")
            else:
                # Transform via est.prep (use full data for fit; sample for SHAP/LIME)
                prep = est.named_steps["prep"]
                X_all = prep.fit_transform(X_raw, y_raw)
                try:
                    feat_names = prep.get_feature_names_out()
                except Exception:
                    feat_names = np.array([f"f{i}" for i in range(X_all.shape[1])])

                # Sampling
                n = X_all.shape[0]
                if n > shap_sample:
                    idxs = np.random.RandomState(RANDOM_STATE).choice(n, shap_sample, replace=False)
                    X_used = X_all[idxs]
                else:
                    X_used = X_all

                model = est.named_steps["model"]

                st.markdown(f"### {cm} — Scenario {cs} — Q{cq}")

                # --- SHAP (TreeExplainer if tree-like; KernelExplainer otherwise, sampled) ---
                shap_values = None
                try:
                    is_tree_like = hasattr(model, "feature_importances_") or "Forest" in model.__class__.__name__
                    if is_tree_like:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_used)
                    else:
                        k_bg_size = min(X_used.shape[0], 200)  # kernel background size cap
                        bg = X_used[:k_bg_size]
                        explainer = shap.KernelExplainer(model.predict, bg)
                        shap_values = explainer.shap_values(X_used, nsamples=300)
                except Exception as e:
                    st.error(f"SHAP failed: {e}")

                # Focus SHAP on CrimeCategory-* features
                if shap_values is not None:
                    mask_cat = np.array([("CrimeCategory" in str(n)) for n in feat_names])
                    if not mask_cat.any():
                        st.info("No explicit CrimeCategory features after encoding; showing all features.")
                        mask_cat = np.ones_like(feat_names, dtype=bool)

                    X_cat = X_used[:, mask_cat]
                    fn_cat = feat_names[mask_cat]

                    # Importance table
                    try:
                        sv = shap_values if isinstance(shap_values, np.ndarray) else np.array(shap_values)
                        mean_abs = np.mean(np.abs(sv), axis=0)
                        mean_abs_cat = mean_abs[mask_cat]
                        shap_df = pd.DataFrame({"CrimeCategoryFeature": fn_cat, "Mean|SHAP|": mean_abs_cat})
                        shap_df.sort_values("Mean|SHAP|", ascending=False, inplace=True)
                    except Exception:
                        shap_df = pd.DataFrame({"CrimeCategoryFeature": fn_cat})

                    # SHAP — Category Importance (bar)
                    st.markdown("**SHAP — Category Importance (bar)**")
                    try:
                        plt.figure(figsize=(7.5, 6.2))
                        shap.summary_plot(sv[:, mask_cat], X_cat, feature_names=fn_cat, plot_type="bar", show=False, max_display=len(fn_cat))
                        ax = plt.gca()
                        ax.set_xlim(left=0)
                        ax.set_xticks([0, 20, 40, 60, 80, 100])
                        style_axes_black(ax)
                        st.pyplot(plt.gcf(), clear_figure=True)
                    except Exception as e:
                        st.info(f"SHAP bar could not render: {e}")

                    # SHAP — Category Beeswarm
                    st.markdown("**SHAP — Category Beeswarm**")
                    try:
                        plt.figure(figsize=(7.5, 6.2))
                        shap.summary_plot(sv[:, mask_cat], X_cat, feature_names=fn_cat, show=False, max_display=len(fn_cat))
                        ax = plt.gca()
                        style_axes_black(ax)
                        st.pyplot(plt.gcf(), clear_figure=True)
                    except Exception as e:
                        st.info(f"SHAP beeswarm could not render: {e}")

                    with st.expander("View SHAP Raw Table (Mean |SHAP| per Category Feature)"):
                        st.dataframe(shap_df.reset_index(drop=True), use_container_width=True)

                # --- LIME (sampled) ---
                st.markdown("**LIME — Local Explanation (sample 1 row from SHAP sample)**")
                try:
                    if X_used.shape[0] == 0:
                        raise ValueError("No rows for LIME.")
                    expl_lime = LimeTabularExplainer(
                        training_data=np.array(X_used),
                        feature_names=list(feat_names),
                        mode="regression"
                    )
                    idx_local = 0
                    exp = expl_lime.explain_instance(np.array(X_used[idx_local]), model.predict, num_features=10)
                    # Render inline
                    st.components.v1.html(exp.as_html(), height=520, scrolling=True)
                except Exception as e:
                    st.info(f"LIME failed: {e}")

                explain_blocks.append((cm, cs, cq, est, prep, X_raw, y_raw))
                st.markdown("---")
                gc.collect()

        # Choose which candidate to accept for forecast (or switch between them for different forecasts)
        if explain_blocks:
            labels_accept = [f"{m} (Scenario {s}) — Q{s_q}" for (m, s, s_q, *_rest) in explain_blocks]
            pick_idx = st.radio("Select a candidate to use for forecasting", options=list(range(len(labels_accept))),
                                format_func=lambda i: labels_accept[i], index=0, key="xai_accept_radio")
            if st.button("✅ Accept Selected Model", key="btn_accept_model"):
                chosen = explain_blocks[pick_idx]
                st.session_state["accepted_model_block"] = {
                    "ModelName": chosen[0],
                    "Scenario":  chosen[1],
                    "Quarter":   chosen[2],
                    "Estimator": chosen[3],   # fitted Pipeline
                    "Preproc":   chosen[4],   # fitted ColumnTransformer
                    "X":         chosen[5],
                    "y":         chosen[6],
                }
                st.success(f"Accepted: {chosen[0]} (Scenario {chosen[1]}) — Q{chosen[2]}")
        else:
            st.info("No candidate blocks to explain. Check training results.")

# ==========================================================
# TAB: Forecast — uses the selected candidate, button-triggered
# ==========================================================
with tab_forecast:
    st.subheader("Forecast (2025–2027) — Uses the Accepted Model (no retrain)")

    acc = st.session_state.get("accepted_model_block")
    if not acc:
        st.info("Accept a model in the SHAP & LIME tab first.")
    else:
        # Choose between the two candidates for this run (without re-accepting)
        top2 = st.session_state.get("top2_overall", pd.DataFrame())
        trained = st.session_state.get("trained_models", {})
        quarters_src = st.session_state.get("out_quarters", st.session_state.get("clean_quarters", quarters))
        if top2.empty:
            st.info("Top-2 list not found; using the accepted model only.")
            candidates = [acc]
            chosen_idx = 0
        else:
            # Build candidate list from top2 for the accepted quarter
            cand_list = []
            for _, r in top2.iterrows():
                est = trained.get((int(r["Scenario"]), int(r["Quarter"]), r["Model"]))
                if est is None: continue
                dfq = quarters_src.get(int(r["Quarter"]), pd.DataFrame())
                if dfq.empty: continue
                ty = latest_year_in(dfq)
                X_raw, y_raw, _ = build_features_by_scenario(dfq, ty, int(r["Scenario"]))
                if X_raw.empty: continue
                cand_list.append({
                    "ModelName": r["Model"],
                    "Scenario":  int(r["Scenario"]),
                    "Quarter":   int(r["Quarter"]),
                    "Estimator": est,
                    "X":         X_raw,
                    "y":         y_raw,
                })
            if not cand_list:
                cand_list = [acc]
            labels = [f"{c['ModelName']} (Scenario {c['Scenario']}) — Q{c['Quarter']}" for c in cand_list]
            chosen_idx = st.radio("Pick candidate for this forecast run", options=list(range(len(labels))),
                                  format_func=lambda i: labels[i], index=0, key="fc_pick_radio")
            candidates = cand_list

        run_forecast = st.button("🔮 Run Forecast with Selected Candidate")
        if run_forecast:
            cand = candidates[chosen_idx]
            cq = cand["Quarter"]
            dfq = quarters_src.get(cq, pd.DataFrame())
            if dfq.empty:
                st.warning("No data for the selected quarter.")
            else:
                dfq = coerce_years_to_numeric(ensure_core_columns(dfq))
                years = detect_year_cols(dfq)
                if not years or len(years) < 2:
                    st.warning("Not enough year columns to forecast.")
                else:
                    # Determine categories to forecast (canonical 8 if present)
                    cats_all = sorted(dfq["CrimeCategory"].dropna().unique().tolist())
                    cats = [c for c in CANON_CATEGORIES if c in cats_all] or cats_all
                    st.caption(f"{len(cats)} categories will be forecasted. Using every year on x-axis with value labels.")

                    from sklearn.ensemble import RandomForestRegressor as RF1D
                    combined_rows = []
                    for cat in cats:
                        sub = dfq[dfq["CrimeCategory"] == cat]
                        if sub.empty: continue
                        series = sub[years].sum(axis=0).astype(float)
                        X_year = np.array([int(y) for y in years]).reshape(-1, 1)
                        y_val = series.values

                        rf = RF1D(n_estimators=250, random_state=RANDOM_STATE)
                        rf.fit(X_year, y_val)

                        fut_years = np.arange(max([int(y) for y in years]) + 1, 2028)
                        if fut_years.size == 0:
                            fut_years = np.array([2025, 2026, 2027])
                        y_future = rf.predict(fut_years.reshape(-1, 1))

                        hist_df = pd.DataFrame({"Year": X_year.flatten(), "Value": y_val, "Type": "Historical", "CrimeCategory": cat})
                        fut_df  = pd.DataFrame({"Year": fut_years, "Value": y_future, "Type": "Forecast", "CrimeCategory": cat})
                        cat_df = pd.concat([hist_df, fut_df], ignore_index=True)
                        combined_rows.append(cat_df)

                        # Per-category chart
                        fig = px.line(
                            cat_df, x="Year", y="Value", color="Type", markers=True,
                            title=f"{cat} — Forecast (Q{cq})"
                        )
                        fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
                        plotly_black(fig)
                        # Annotate all values
                        for _, r in cat_df.iterrows():
                            fig.add_annotation(x=r["Year"], y=r["Value"], text=f"{r['Value']:.0f}",
                                               showarrow=False, yanchor="bottom", font=dict(size=10, color="black"))
                        st.plotly_chart(fig, use_container_width=True)

                    if combined_rows:
                        all_df = pd.concat(combined_rows, ignore_index=True)
                        st.markdown("### Combined — All Categories")
                        figc = px.line(
                            all_df, x="Year", y="Value", color="CrimeCategory", line_dash="Type", markers=True,
                            title=f"All Categories — Combined Forecast (Q{cq})"
                        )
                        figc.update_layout(xaxis=dict(tickmode="linear", dtick=1))
                        plotly_black(figc)
                        st.plotly_chart(figc, use_container_width=True)

# ==========================================================
# TAB: Compare Stations — single category chart + all-categories-at-once
# ==========================================================
with tab_compare:
    st.subheader("Compare Stations (uses accepted model's quarter automatically if set)")
    acc = st.session_state.get("accepted_model_block")
    # Use accepted quarter if available, else default to Q1
    default_q = acc["Quarter"] if acc else 1
    cq = st.selectbox("Quarter for comparison", [1,2,3,4], index=[1,2,3,4].index(default_q), key="cmp_quarter")

    dfq = st.session_state.get("out_quarters", st.session_state.get("clean_quarters", quarters)).get(cq, pd.DataFrame())
    if dfq.empty:
        st.warning("No data for selected quarter.")
    else:
        dfq = ensure_core_columns(coerce_years_to_numeric(dfq))
        years = detect_year_cols(dfq)
        if not years:
            st.warning("No years found for comparison.")
        else:
            stations_all = dfq["PoliceStationCode"].dropna().unique().tolist()
            labels_map = {code: station_code_to_label(df_stations, code) for code in stations_all}
            label_to_code = {v: k for k, v in labels_map.items()}
            pick_labels = st.multiselect(
                "Pick stations to compare (2–6 recommended)",
                sorted(labels_map.values()),
                default=sorted(labels_map.values())[:2],
                key="cmp_station_mult"
            )
            if not pick_labels:
                st.info("Select at least one station.")
            else:
                cat_for_cmp = st.selectbox(
                    "Crime Category",
                    options=[c for c in CANON_CATEGORIES if c in dfq["CrimeCategory"].unique()] or sorted(dfq["CrimeCategory"].unique()),
                    index=0,
                    key="cmp_category_sel"
                )
                # Single-category comparison
                rows = []
                from sklearn.ensemble import RandomForestRegressor as RF1D
                for lab in pick_labels:
                    code = label_to_code.get(lab, lab.split(" - ")[0])
                    sub = dfq[(dfq["PoliceStationCode"] == code) & (dfq["CrimeCategory"] == cat_for_cmp)]
                    if sub.empty:
                        continue
                    ser = sub[years].iloc[0].astype(float)
                    Xy = np.array([int(y) for y in years]).reshape(-1,1)
                    vy = ser.values
                    rf = RF1D(n_estimators=200, random_state=RANDOM_STATE)
                    rf.fit(Xy, vy)
                    fy = np.arange(max([int(y) for y in years]) + 1, 2028)
                    if fy.size == 0: fy = np.array([2025, 2026, 2027])
                    pv = rf.predict(fy.reshape(-1,1))
                    his_df = pd.DataFrame({"Year": Xy.flatten(), "Value": vy, "Type": "Historical", "Station": lab})
                    fut_df = pd.DataFrame({"Year": fy, "Value": pv, "Type": "Forecast", "Station": lab})
                    rows.append(pd.concat([his_df, fut_df], ignore_index=True))
                if rows:
                    comp = pd.concat(rows, ignore_index=True)
                    fig = px.line(comp, x="Year", y="Value", color="Station", line_dash="Type", markers=True,
                                  title=f"Station Comparison — {cat_for_cmp} (Q{cq})")
                    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
                    plotly_black(fig)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No comparable station data found.")

                # --- All-categories-at-once comparison for the selected stations ---
                st.markdown("### All Categories — Station Comparison (Historical + Forecast)")
                categories_available = dfq["CrimeCategory"].dropna().unique().tolist()
                all_cats = [c for c in CANON_CATEGORIES if c in categories_available] or sorted(categories_available)
                rows_allcats = []
                for lab in pick_labels:
                    st_code = label_to_code.get(lab, lab.split(" - ")[0])
                    for cat in all_cats:
                        sub = dfq[(dfq["PoliceStationCode"] == st_code) & (dfq["CrimeCategory"] == cat)]
                        if sub.empty: continue
                        series = sub[years].iloc[0].astype(float)
                        Xy = np.array([int(y) for y in years]).reshape(-1, 1)
                        vy = series.values
                        rf_all = RF1D(n_estimators=200, random_state=RANDOM_STATE)
                        rf_all.fit(Xy, vy)
                        fy = np.arange(max([int(y) for y in years]) + 1, 2028)
                        if fy.size == 0: fy = np.array([2025, 2026, 2027])
                        pv = rf_all.predict(fy.reshape(-1, 1))
                        his_df = pd.DataFrame({"Year": Xy.flatten(), "Value": vy, "Type": "Historical", "Station": lab, "CrimeCategory": cat})
                        fut_df = pd.DataFrame({"Year": fy, "Value": pv, "Type": "Forecast", "Station": lab, "CrimeCategory": cat})
                        rows_allcats.append(pd.concat([his_df, fut_df], ignore_index=True))
                if rows_allcats:
                    allcats_df = pd.concat(rows_allcats, ignore_index=True)
                    fig_allcats = px.line(
                        allcats_df,
                        x="Year", y="Value",
                        color="CrimeCategory",
                        line_dash="Type",
                        facet_col="Station",
                        facet_col_wrap=2,
                        markers=True,
                        title=f"All Categories — Comparison across Stations (Q{cq})"
                    )
                    fig_allcats.update_layout(
                        xaxis=dict(tickmode="linear", dtick=1),
                        legend_title_text="CrimeCategory",
                        margin=dict(t=60, r=10, b=10, l=10),
                    )
                    plotly_black(fig_allcats)
                    st.plotly_chart(fig_allcats, use_container_width=True)

                    # Optional: latest-year snapshot
                    latest_year = max([int(y) for y in years])
                    latest_rows = []
                    for lab in pick_labels:
                        code = label_to_code.get(lab, lab.split(" - ")[0])
                        for cat in all_cats:
                            sub = dfq[(dfq["PoliceStationCode"] == code) & (dfq["CrimeCategory"] == cat)]
                            if sub.empty: continue
                            val = float(sub[str(latest_year)].sum())
                            latest_rows.append({"Station": lab, "CrimeCategory": cat, "Year": latest_year, "Value": val})
                    if latest_rows:
                        latest_df = pd.DataFrame(latest_rows)
                        fig_bar = px.bar(
                            latest_df, x="Station", y="Value", color="CrimeCategory",
                            barmode="group",
                            title=f"Latest Historical Year ({latest_year}) — Totals by Station & Category"
                        )
                        plotly_black(fig_bar)
                        fig_bar.update_layout(xaxis_title="", yaxis_title="Value", margin=dict(t=60, r=10, b=10, l=10))
                        st.plotly_chart(fig_bar, use_container_width=True)
