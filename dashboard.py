"""
Interactive Data Mining Dashboard - Road Traffic Accidents (RTA)
================================================================
Companion of rta_data_mining.py. It does NOT predict accidents: it only shows the patterns
that were discovered (frequent patterns, association rules, clusters, outliers).

Run order:
    pip install -r requirements.txt
    python rta_data_mining.py          # once -> creates the rta_results/ folder
    streamlit run dashboard.py

Data sources
    * Default : the CSV files saved in rta_results/ by rta_data_mining.py
    * Upload  : any other CSV (sidebar). Expected columns are detected automatically, analyses whose
                columns are missing are skipped with a warning, and rules / clusters / outliers are
                computed live with the same methods used in rta_data_mining.py.
"""

import inspect
import io
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="RTA Data Mining Dashboard", page_icon="🚦", layout="wide")

st.markdown(
    """
<style>
.banner{background:linear-gradient(90deg,#6a11cb 0%,#2575fc 50%,#00c9a7 100%);padding:22px 28px;
        border-radius:16px;color:white;margin-bottom:14px}
.banner h1{margin:0;font-size:2rem;color:white}
.banner p{margin:6px 0 0 0;opacity:.95;color:white}
.kpi{border-radius:16px;padding:16px 14px;color:white;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.18)}
.kpi .v{font-size:2rem;font-weight:800;line-height:1.15}
.kpi .l{font-size:.9rem;opacity:.95}
.explain{background:#f3f6ff;border-left:6px solid #2575fc;padding:10px 14px;border-radius:8px;
         color:#1b2a4a;margin:6px 0 14px 0}
</style>
<div class="banner"><h1>🚦 Road Traffic Accident - Data Mining Dashboard</h1>
<p>Discovering hidden patterns with frequent pattern mining, association rules (Apriori),
K-Means clustering and outlier detection. This dashboard describes patterns - it does not predict accidents.</p></div>
""",
    unsafe_allow_html=True,
)

# =====================================================================================
# CONSTANTS
# =====================================================================================
SEED = 42
SEV = "Accident_severity"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PERIOD_ORDER = ["Night (00-05)", "Morning (06-11)", "Afternoon (12-17)", "Evening (18-23)"]

EXPECTED_CAT = [
    "Day_of_week", "Age_band_of_driver", "Sex_of_driver", "Educational_level", "Vehicle_driver_relation",
    "Driving_experience", "Type_of_vehicle", "Owner_of_vehicle", "Service_year_of_vehicle", "Defect_of_vehicle",
    "Area_accident_occured", "Lanes_or_Medians", "Road_allignment", "Types_of_Junction", "Road_surface_type",
    "Road_surface_conditions", "Light_conditions", "Weather_conditions", "Type_of_collision", "Vehicle_movement",
    "Casualty_class", "Sex_of_casualty", "Age_band_of_casualty", "Casualty_severity", "Work_of_casuality",
    "Fitness_of_casuality", "Pedestrian_movement", "Cause_of_accident", "Accident_severity",
    "Time_period", "Vehicles_involved", "Casualties",          # the last three are derived columns
]
EXPECTED_NUM = ["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]
CLUSTER_FEATURES = [
    "Age_band_of_driver", "Driving_experience", "Type_of_vehicle", "Area_accident_occured", "Lanes_or_Medians",
    "Types_of_Junction", "Light_conditions", "Road_surface_conditions", "Type_of_collision", "Vehicle_movement",
    "Cause_of_accident", "Time_period",
]
CLUSTER_NUMS = ["Number_of_vehicles_involved", "Number_of_casualties"]
LABEL_COLS = ["Cluster_best_silhouette", "Cluster_k4_finer_view", "Anomaly_score", "Outlier"]

SAVED_FILES = {
    "cleaned": "rta_cleaned.csv",
    "with_labels": "rta_with_clusters_and_outliers.csv",
    "rules_general": "association_rules_general.csv",
    "rules_specific": "association_rules_specific_support2pct.csv",
    "rules_severity": "association_rules_severity_support2pct.csv",
    "rules_fatal": "association_rules_fatal.csv",
    "rules_serious": "association_rules_serious.csv",
    "patterns": "frequent_itemsets.csv",
    "profile_best": "cluster_profiles_best_silhouette.csv",
    "profile_k4": "cluster_profiles_k4_finer_view.csv",
    "outlier_items": "outlier_overrepresented_items.csv",
    "feature_assoc": "feature_association_with_severity.csv",
    "severity_rate": "severity_rate_by_category.csv",
}
RULE_SETS = {
    "rules_general": "General rules (support ≥ 5%, confidence ≥ 60%, lift ≥ 1.2)",
    "rules_specific": "More specific rules (support ≥ 2%, confidence ≥ 50%, lift ≥ 1.5)",
    "rules_severity": "Rules that end in an accident-severity level (support ≥ 2%)",
    "rules_fatal": "Patterns linked to FATAL accidents (mined inside the fatal group)",
    "rules_serious": "Patterns linked to SERIOUS accidents (mined inside the serious group)",
}

VIVID = px.colors.qualitative.Vivid
BOLD = px.colors.qualitative.Bold
SET2 = px.colors.qualitative.Set2


# =====================================================================================
# SMALL HELPERS
# =====================================================================================
def explain(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def _stretch_kw(func):
    """Full-width argument that works on old and new Streamlit versions."""
    try:
        p = inspect.signature(func).parameters.get("width")
        if p is not None and p.default == "stretch":
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def plot(fig, key):
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, key=key, **_stretch_kw(st.plotly_chart))


def show_df(df, column_config=None, height=None):
    kw = dict(hide_index=True)
    if column_config:
        kw["column_config"] = column_config
    if height:
        kw["height"] = height
    st.dataframe(df, **_stretch_kw(st.dataframe), **kw)


def short(text, n=60):
    text = str(text).replace("_", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


def counts(series, name):
    out = series.value_counts().reset_index()
    out.columns = [name, "Accidents"]
    return out


def sev_color_map(values):
    m = {}
    for v in pd.unique(pd.Series(values).dropna()):
        s = str(v).lower()
        if any(k in s for k in ("slight", "minor")):
            m[v] = "#2ECC71"
        elif any(k in s for k in ("serious", "severe")):
            m[v] = "#FF9F1C"
        elif any(k in s for k in ("fatal", "death", "killed")):
            m[v] = "#E63946"
    return m


def is_text(s):
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s))


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


CANON = {_norm(c): c for c in EXPECTED_CAT + EXPECTED_NUM + ["Time"]}
CANON.update({
    "severity": "Accident_severity", "weather": "Weather_conditions", "weathercondition": "Weather_conditions",
    "roadcondition": "Road_surface_conditions", "roadconditions": "Road_surface_conditions",
    "roadsurfacecondition": "Road_surface_conditions", "lightcondition": "Light_conditions",
    "day": "Day_of_week", "weekday": "Day_of_week", "cause": "Cause_of_accident",
    "accidentcause": "Cause_of_accident", "roadalignment": "Road_allignment",
    "areaaccidentoccurred": "Area_accident_occured",
})


def canonicalize(df):
    """Rename columns that match an expected column (ignoring case, spaces, underscores)."""
    rename, taken = {}, set(df.columns)
    for c in df.columns:
        target = CANON.get(_norm(c))
        if target and target != c and target not in taken and target not in rename.values():
            rename[c] = target
    return df.rename(columns=rename), rename


def ensure_types(df):
    df = df.copy()
    for c in EXPECTED_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Vehicles_involved", "Casualties"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: x if pd.isna(x) else str(x))
    return df


MISSING_TOKENS = {"na", "n/a", "nan", "null", "unknown", "unknown or other", "?", ""}


def _parse_hour(series):
    t = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    if t.isna().mean() > 0.5:
        t = pd.to_datetime(series, errors="coerce")
    return t.dt.hour


def clean_uploaded(df):
    """Same cleaning ideas as rta_data_mining.py, applied only when the column exists."""
    notes = []
    df = df.copy()
    for c in df.columns:                                   # text: strip spaces, unknown -> missing
        if is_text(df[c]):
            s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            bad = s.str.lower().isin(MISSING_TOKENS).fillna(False).astype(bool)
            s = s.mask(bad)
            s = s.str.replace("statioNot a Pedestrianry", "stationary", regex=False)
            s = s.mask(s == "NormalNormal", "Normal").mask(s == "other", "Other")
            df[c] = s.astype(object).where(s.notna(), np.nan)
    if "Defect_of_vehicle" in df.columns:
        df.loc[df["Defect_of_vehicle"].isin(["5", "7"]), "Defect_of_vehicle"] = np.nan
    if "Age_band_of_casualty" in df.columns:
        df.loc[df["Age_band_of_casualty"] == "5", "Age_band_of_casualty"] = np.nan
    if "Casualty_severity" in df.columns:
        v = df["Casualty_severity"].astype(str).str.replace(r"\.0$", "", regex=True)
        df["Casualty_severity"] = v.map({"1": "Level 1", "2": "Level 2", "3": "Level 3"})
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed {n_dup} duplicate rows.")
    for c in EXPECTED_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Hour" not in df.columns and "Time" in df.columns:
        df["Hour"] = _parse_hour(df["Time"])
        notes.append("Created `Hour` from `Time`.")
    if "Hour" in df.columns and "Time_period" not in df.columns:
        df["Time_period"] = pd.cut(df["Hour"], bins=[-1, 5, 11, 17, 23], labels=PERIOD_ORDER).astype(object)
        notes.append("Created `Time_period` from `Hour`.")
    for src, new in [("Number_of_vehicles_involved", "Vehicles_involved"), ("Number_of_casualties", "Casualties")]:
        if src in df.columns and new not in df.columns:
            df[new] = df[src].map(lambda x: np.nan if pd.isna(x) else ("3+" if x >= 3 else str(int(x))))
    return ensure_types(df), notes


@st.cache_data(show_spinner="Reading and cleaning the uploaded file...")
def prepare_upload(raw, name):
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            if df.shape[1] == 1:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return None, [], {}
    df, renames = canonicalize(df)
    df, notes = clean_uploaded(df)
    return df, notes, renames


@st.cache_data(show_spinner=False)
def read_csv_cached(path, mtime):
    return pd.read_csv(path)


def load_saved(res_dir):
    out = {}
    for key, fn in SAVED_FILES.items():
        p = os.path.join(res_dir, fn)
        if os.path.exists(p):
            out[key] = read_csv_cached(p, os.path.getmtime(p))
    return out


# =====================================================================================
# DATA-MINING FUNCTIONS (same logic as rta_data_mining.py)
# =====================================================================================
def mining_columns(df):
    """Categorical columns usable for pattern mining (<=40% missing, no single value above 95%)."""
    cols = []
    for c in EXPECTED_CAT:
        if c in df.columns and is_text(df[c]):
            s = df[c]
            if s.isna().mean() > 0.40:
                continue
            vc = s.value_counts(normalize=True)
            if len(vc) < 2 or vc.iloc[0] > 0.95:
                continue
            cols.append(c)
    return cols


def onehot_of(d, cols):
    t = d[cols].astype(str).where(d[cols].notna())
    return pd.get_dummies(t, columns=list(cols), prefix_sep="=", dtype=bool)


def apriori(data, min_support, max_len=3):
    """Level-wise Apriori. Returns {frozenset(items): support}."""
    cols = {c: data[c].values for c in data.columns}
    freq, current = {}, []
    for c, arr in cols.items():
        s = arr.mean()
        if s >= min_support:
            freq[frozenset([c])] = s
            current.append((c,))
    current.sort()
    k = 2
    while current and k <= max_len:
        prev = set(current)
        by_prefix = {}
        for itemset in current:
            by_prefix.setdefault(itemset[:-1], []).append(itemset[-1])
        nxt = []
        for prefix, lasts in by_prefix.items():
            for a, b in combinations(sorted(lasts), 2):
                cand = prefix + (a, b)
                if k > 2 and not all(sub in prev for sub in combinations(cand, k - 1)):
                    continue
                s = np.logical_and.reduce([cols[i] for i in cand]).mean()
                if s >= min_support:
                    freq[frozenset(cand)] = s
                    nxt.append(cand)
        current = sorted(nxt)
        k += 1
    return freq


def make_rules(freq, min_conf, min_lift, min_improvement=0.02):
    """Rules with ONE item in the consequent; redundant rules are removed."""
    rows = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        for c in itemset:
            ante = itemset - {c}
            conf = sup / freq[ante]
            lift = conf / freq[frozenset([c])]
            if conf < min_conf or lift < min_lift:
                continue
            if len(ante) >= 2:
                best_sub = max(freq[sub | {c}] / freq[sub]
                               for r in range(1, len(ante))
                               for sub in map(frozenset, combinations(ante, r)))
                if conf - best_sub < min_improvement:
                    continue
            rows.append((" AND ".join(sorted(ante)), c, sup, conf, lift, len(itemset)))
    out = pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift", "size"])
    return out.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Mining frequent patterns with Apriori...")
def mine_freq(_df, key, min_support, max_len, drop_common):
    cols = mining_columns(_df)
    oh = onehot_of(_df, cols)
    sev_items = [c for c in oh.columns if c.startswith(SEV + "=")]
    if drop_common:
        common = oh.columns[(oh.mean() > 0.80) & ~oh.columns.isin(sev_items)]
        oh = oh.drop(columns=common)
    return apriori(oh, min_support, max_len), len(oh)


@st.cache_data(show_spinner=False)
def mine_rules(_freq, key, min_conf, min_lift):
    return make_rules(_freq, min_conf, min_lift)


def build_cluster_matrix(df):
    """One-hot categorical features + scaled numeric features (same recipe as rta_data_mining.py)."""
    cols = [c for c in CLUSTER_FEATURES if c in df.columns and df[c].notna().any()]
    if len(cols) < 3:
        return None, None
    cdf = df[cols].astype(str).where(df[cols].notna())
    for c in cols:
        cdf[c] = cdf[c].fillna(cdf[c].mode()[0])
        vc = cdf[c].value_counts(normalize=True)
        cdf[c] = cdf[c].where(~cdf[c].isin(vc[vc < 0.02].index), "Rare")
    Xc = pd.get_dummies(cdf, columns=cols, prefix_sep="=", dtype=float)
    cat_cols = list(Xc.columns)
    for c in CLUSTER_NUMS:
        if c in df.columns and df[c].notna().any():
            v = pd.to_numeric(df[c], errors="coerce").clip(upper=4)
            v = v.fillna(v.median())
            rng = v.max() - v.min()
            Xc[c] = (v - v.min()) / rng if rng > 0 else 0.0
    return Xc, cat_cols


@st.cache_resource(show_spinner="Preparing features for clustering...")
def get_Xc(_df, key):
    return build_cluster_matrix(_df)


@st.cache_data(show_spinner="Computing the 2-D PCA view...")
def get_pca(_df, key):
    Xc, _ = get_Xc(_df, key)
    if Xc is None:
        return None, None
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(Xc)
    return pd.DataFrame(Z, index=_df.index, columns=["PC1", "PC2"]), float(pca.explained_variance_ratio_.sum())


@st.cache_data(show_spinner="Running K-Means...")
def run_kmeans(_df, key, k):
    Xc, _ = get_Xc(_df, key)
    return KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(Xc)


def _silhouette(Xc, labels, n=3000):
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Xc), min(n, len(Xc)), replace=False)
    if len(set(labels[idx])) < 2:
        return float("nan")
    return float(silhouette_score(Xc.values[idx], labels[idx]))


@st.cache_data(show_spinner=False)
def silhouette_of(_df, key, labels_key, _labels):
    Xc, _ = get_Xc(_df, key)
    return _silhouette(Xc, np.asarray(_labels))


@st.cache_data(show_spinner="Trying k = 2 ... 8 (a few seconds)...")
def silhouette_table(_df, key):
    Xc, _ = get_Xc(_df, key)
    rows = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit_predict(Xc)
        rows.append((k, _silhouette(Xc, labels)))
    return pd.DataFrame(rows, columns=["k", "silhouette"])


@st.cache_data(show_spinner="Running Isolation Forest...")
def run_iforest(_df, key, contamination):
    cols = [c for c in mining_columns(_df) if c != SEV]
    X = onehot_of(_df, cols).astype(float)
    for c in CLUSTER_NUMS:
        if c in _df.columns and _df[c].notna().any():
            v = pd.to_numeric(_df[c], errors="coerce").clip(upper=4)
            X[c] = (v.fillna(v.median()) - 1) / 3
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=SEED).fit(X)
    return -iso.score_samples(X), iso.predict(X) == -1


def cluster_profile(Xcat, labels, min_share=0.30, min_lift=1.15, top=8):
    overall = Xcat.mean()
    rows = []
    for cl in sorted(labels.unique()):
        m = (labels == cl).values
        share = Xcat[m].mean()
        lift = share / overall
        sel = pd.DataFrame({"share_in_cluster": share, "share_overall": overall, "lift": lift})
        sel = sel[(sel["share_in_cluster"] >= min_share) & (sel["lift"] >= min_lift)]
        for item, r in sel.sort_values("lift", ascending=False).head(top).iterrows():
            rows.append((cl, item, r["share_in_cluster"], r["share_overall"], r["lift"]))
    return pd.DataFrame(rows, columns=["cluster", "category", "share_in_cluster", "share_overall", "lift"])


def overrepresented(d, flag, cols, min_count):
    X = onehot_of(d, cols)
    f = X[flag.values]
    share_out, share_all = f.mean(), X.mean()
    out = pd.DataFrame({
        "category": X.columns,
        "share_in_outliers": share_out.values,
        "share_overall": share_all.values,
        "lift": (share_out / share_all).values,
        "outlier_count": f.sum().values,
    })
    return out[out["outlier_count"] >= min_count].sort_values("lift", ascending=False).reset_index(drop=True)


def ordered_options(s, col):
    vals = list(s.dropna().unique())
    order = DAY_ORDER if col == "Day_of_week" else PERIOD_ORDER if col == "Time_period" else None
    if order:
        return [o for o in order if o in vals] + sorted([v for v in vals if v not in order], key=str)
    return list(s.value_counts().index)


# =====================================================================================
# SIDEBAR - 1. DATA SOURCE
# =====================================================================================
st.sidebar.title("🚦 RTA Data Mining")
st.sidebar.header("1️⃣ Data source")
uploaded = st.sidebar.file_uploader(
    "Upload another CSV dataset (optional)", type=["csv"],
    help="Use RTA-style column names (e.g. Day_of_week, Weather_conditions, Accident_severity). "
         "Missing columns are fine - the analyses that need them are skipped with a warning.")
res_dir = st.sidebar.text_input("Results folder (from rta_data_mining.py)", value="rta_results")

saved = {}
if uploaded is not None:
    mode = "upload"
    base, upload_notes, renames = prepare_upload(uploaded.getvalue(), uploaded.name)
    if base is None:
        st.error("❌ The uploaded file could not be read as a CSV table. Please check the file.")
        st.stop()
    data_key = f"upload|{uploaded.name}|{uploaded.size}"
    st.sidebar.success(f"Using uploaded file: {uploaded.name} ({len(base):,} rows × {base.shape[1]} columns)")
    with st.sidebar.expander("What was detected / cleaned"):
        if renames:
            st.write("Columns renamed to the expected names:", {k: v for k, v in renames.items()})
        for n in upload_notes:
            st.write("•", n)
        st.write("Missing/unknown values (NA, Unknown, ?) were treated as missing.")
else:
    mode = "project"
    saved = load_saved(res_dir)
    if "cleaned" not in saved:
        st.error(f"❌ `{os.path.join(res_dir, SAVED_FILES['cleaned'])}` was not found. Run "
                 "`python rta_data_mining.py` first (it creates the rta_results/ folder), "
                 "change the results folder in the sidebar, or upload a CSV.")
        st.stop()
    base = ensure_types(saved["cleaned"])
    lab = saved.get("with_labels")
    if lab is not None and len(lab) == len(base):
        for c in LABEL_COLS:
            if c in lab.columns:
                base[c] = lab[c].values
        if "Outlier" in base.columns:
            base["Outlier"] = base["Outlier"].astype(str).str.lower().eq("true")
    data_key = f"project|{res_dir}|{os.path.getmtime(os.path.join(res_dir, SAVED_FILES['cleaned']))}"
    st.sidebar.success(f"Loaded project results from `{res_dir}/`")
    missing_files = [fn for k, fn in SAVED_FILES.items() if k not in saved]
    if missing_files:
        with st.sidebar.expander("Result files not found"):
            st.write(", ".join(missing_files))
            st.caption("The related sections fall back to live computation where possible.")


def has(col, d=None):
    d = base if d is None else d
    return col in d.columns and bool(d[col].notna().any())


def require(cols, name):
    missing = [c for c in cols if not has(c)]
    if missing:
        names = ", ".join("`Hour` (or a `Time` column)" if c == "Hour" else f"`{c}`" for c in missing)
        st.warning(f"⚠️ **{name}** is skipped because the required column(s) are missing or empty: {names}")
        return False
    return True


# =====================================================================================
# SIDEBAR - 2. FILTERS
# =====================================================================================
st.sidebar.header("2️⃣ Filters")
FILTERS = [("Accident severity", SEV), ("Weather", "Weather_conditions"), ("Road condition", "Road_surface_conditions"),
           ("Light condition", "Light_conditions"), ("Day of week", "Day_of_week"), ("Time period", "Time_period")]
mask = pd.Series(True, index=base.index)
sig = []
for label, col in FILTERS:
    if has(col):
        opts = ordered_options(base[col], col)
        sel = st.sidebar.multiselect(label, opts, default=opts, key=f"f_{col}_{data_key}")
        if set(sel) != set(opts):
            mask &= base[col].isin(sel)
            sig.append((col, tuple(sorted(map(str, sel)))))
    else:
        st.sidebar.caption(f"⚠️ **{label}** filter unavailable - column `{col}` not found.")
fdf = base[mask]
fkey = data_key + "|" + str(sig)
st.sidebar.caption(f"**{len(fdf):,}** of {len(base):,} accidents selected")
if fdf.empty:
    st.warning("No accidents match the current filters. Please select more options in the sidebar.")
    st.stop()

# =====================================================================================
# SIDEBAR - 3. MINING SETTINGS
# =====================================================================================
st.sidebar.header("3️⃣ Pattern-mining settings")
saved_rules_ok = mode == "project" and any(k in saved for k in RULE_SETS)
if saved_rules_ok:
    src = st.sidebar.radio("Rules & patterns source", ["Saved project results", "Mine live on filtered data"],
                           key="mining_source")
    use_saved = src.startswith("Saved")
else:
    use_saved = False
    st.sidebar.caption("Rules and patterns are mined live (Apriori) from the selected accidents.")
ignore_common = st.sidebar.checkbox("Ignore very common items (present in >80% of accidents)", value=True,
                                    help="Items such as 'Male driver' or 'Asphalt road' appear almost everywhere, "
                                         "so they create boring patterns.")
min_sup, min_conf, min_lift, max_len = 0.05, 0.60, 1.2, 3
if not use_saved:
    min_sup = st.sidebar.slider("Minimum support", 0.02, 0.30, 0.05, 0.01)
    min_conf = st.sidebar.slider("Minimum confidence", 0.30, 1.00, 0.60, 0.05)
    min_lift = st.sidebar.slider("Minimum lift", 1.0, 5.0, 1.2, 0.1)
    max_len = st.sidebar.slider("Max items per pattern", 2, 4, 3, help="4 is slower.")

# =====================================================================================
# KPI CARDS + KEY FINDINGS
# =====================================================================================
sev_lower = fdf[SEV].astype(str).str.lower() if has(SEV) else None


def kpi(col, label, value, sub, c1, c2):
    col.markdown(f'<div class="kpi" style="background:linear-gradient(135deg,{c1},{c2})"><div class="l">{label}</div>'
                 f'<div class="v">{value}</div><div class="l">{sub}</div></div>', unsafe_allow_html=True)


k1, k2, k3, k4, k5 = st.columns(5)
n_all = len(fdf)
kpi(k1, "Total accidents", f"{n_all:,}", f"{n_all / len(base) * 100:.0f}% of the dataset", "#6a11cb", "#2575fc")
if sev_lower is not None:
    n_slight = int(sev_lower.str.contains("slight|minor").sum())
    n_serious = int(sev_lower.str.contains("serious|severe").sum())
    n_fatal = int(sev_lower.str.contains("fatal|death|killed").sum())
    kpi(k2, "Slight injuries", f"{n_slight:,}", f"{n_slight / n_all:.1%}", "#11998e", "#38ef7d")
    kpi(k3, "Serious injuries", f"{n_serious:,}", f"{n_serious / n_all:.1%}", "#f46b45", "#eea849")
    kpi(k4, "Fatal accidents", f"{n_fatal:,}", f"{n_fatal / n_all:.1%}", "#b31217", "#e52d27")
else:
    for c, lbl in [(k2, "Slight injuries"), (k3, "Serious injuries"), (k4, "Fatal accidents")]:
        kpi(c, lbl, "N/A", "column missing", "#7f8c8d", "#95a5a6")
if has("Number_of_casualties"):
    kpi(k5, "Average casualties", f"{fdf['Number_of_casualties'].mean():.2f}", "per accident", "#8E2DE2", "#4A00E0")
else:
    kpi(k5, "Average casualties", "N/A", "column missing", "#7f8c8d", "#95a5a6")
if sev_lower is None:
    st.warning(f"⚠️ Severity KPIs are skipped because the required column `{SEV}` is missing or empty.")
elif n_slight + n_serious + n_fatal == 0:
    st.warning("⚠️ The severity labels were not recognised (expected words like Slight / Serious / Fatal), "
               "so the severity KPIs show 0.")
if not has("Number_of_casualties"):
    st.warning("⚠️ **Average casualties** is skipped because the required column `Number_of_casualties` is missing.")

findings = []
if has("Hour"):
    h = fdf["Hour"].dropna()
    if len(h):
        hc = h.astype(int).value_counts()
        findings.append(f"⏰ Busiest hour: **{hc.idxmax()}:00** ({hc.max():,} accidents).")
if has("Day_of_week"):
    dc = fdf["Day_of_week"].value_counts()
    findings.append(f"📅 Day with most accidents: **{dc.idxmax()}** ({dc.max():,}).")
if has("Cause_of_accident"):
    cc = fdf["Cause_of_accident"].value_counts()
    findings.append(f"⚠️ Most common cause: **{cc.idxmax()}** ({cc.max() / n_all:.1%} of accidents).")
if sev_lower is not None and n_slight + n_serious + n_fatal > 0:
    findings.append(f"🚑 **{(n_serious + n_fatal) / n_all:.1%}** of the selected accidents were serious or fatal.")
if findings:
    st.info("**💡 Key findings for the current selection**\n\n" + "\n\n".join(f"- {f}" for f in findings))

# =====================================================================================
# TABS
# =====================================================================================
tab_over, tab_fp, tab_rules, tab_cl, tab_out, tab_data = st.tabs(
    ["📊 Overview", "🧩 Frequent Patterns", "🔗 Association Rules", "🧬 Clustering", "🚨 Outlier Detection",
     "📁 Data & About"])

# ---------------------------------------------------------------- OVERVIEW
with tab_over:
    st.header("📊 Overview of the accidents")
    explain("These charts describe <b>what happened</b> in the accidents that match your sidebar filters. "
            "Use the filters to compare, for example, rainy days with dry days.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accident severity")
        if require([SEV], "Severity distribution"):
            vc = counts(fdf[SEV].dropna(), SEV)
            fig = px.pie(vc, names=SEV, values="Accidents", hole=0.5, color=SEV,
                         color_discrete_map=sev_color_map(vc[SEV]), color_discrete_sequence=VIVID)
            plot(fig, "ov_sev")
    with c2:
        st.subheader("Accidents by hour of the day")
        if require(["Hour"], "Accidents by hour"):
            h = fdf["Hour"].dropna()
            h = h[(h >= 0) & (h <= 23)].astype(int)
            full = pd.DataFrame({"Hour": range(24)}).merge(counts(h, "Hour"), on="Hour", how="left").fillna(0)
            fig = px.bar(full, x="Hour", y="Accidents", color="Accidents", color_continuous_scale="Turbo")
            fig.update_xaxes(dtick=1)
            plot(fig, "ov_hour")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Accidents by day of the week")
        if require(["Day_of_week"], "Accidents by day"):
            dc = counts(fdf["Day_of_week"].dropna(), "Day_of_week")
            fig = px.bar(dc, x="Day_of_week", y="Accidents", color="Day_of_week", color_discrete_sequence=VIVID,
                         category_orders={"Day_of_week": DAY_ORDER})
            fig.update_layout(showlegend=False)
            plot(fig, "ov_day")
    with c4:
        st.subheader("Top accident causes")
        if require(["Cause_of_accident"], "Top accident causes"):
            n_causes = st.slider("Number of causes to show", 5, 20, 10, key="n_causes")
            cc = counts(fdf["Cause_of_accident"].dropna(), "Cause").head(n_causes).sort_values("Accidents")
            fig = px.bar(cc, x="Accidents", y="Cause", orientation="h", color="Accidents",
                         color_continuous_scale="Sunset")
            plot(fig, "ov_cause")

    st.subheader("How do weather and road condition relate to severity?")
    explain("Each bar is split by severity. Choose <b>% within each category</b> to compare conditions fairly, "
            "because some conditions (like normal weather) have many more accidents than others.")
    view = st.radio("Show as", ["Number of accidents", "% within each category"], horizontal=True, key="ov_view")
    percent = view.startswith("%")

    def severity_by(col, title, key):
        ct = fdf.dropna(subset=[col, SEV]).groupby([col, SEV]).size().reset_index(name="Accidents")
        if ct.empty:
            st.info("No data for this chart with the current filters.")
            return
        y = "Accidents"
        if percent:
            ct["Share (%)"] = ct["Accidents"] / ct.groupby(col)["Accidents"].transform("sum") * 100
            y = "Share (%)"
        order = ct.groupby(col)["Accidents"].sum().sort_values(ascending=False).index.tolist()
        fig = px.bar(ct, x=col, y=y, color=SEV, barmode="stack", title=title, category_orders={col: order},
                     color_discrete_map=sev_color_map(ct[SEV]), color_discrete_sequence=VIVID)
        plot(fig, key)

    c5, c6 = st.columns(2)
    with c5:
        if require(["Weather_conditions", SEV], "Weather vs severity"):
            severity_by("Weather_conditions", "Weather vs severity", "ov_weather")
    with c6:
        if require(["Road_surface_conditions", SEV], "Road condition vs severity"):
            severity_by("Road_surface_conditions", "Road condition vs severity", "ov_road")

    if mode == "project" and ("feature_assoc" in saved or "severity_rate" in saved):
        with st.expander("🔎 Which factors are linked to severity? (saved analysis from rta_results/)"):
            if "feature_assoc" in saved:
                fa = saved["feature_assoc"].head(12).copy()
                fa["Significant (p<0.05)"] = np.where(fa["p_value"] < 0.05, "Yes", "No")
                fig = px.bar(fa.sort_values("cramers_v"), x="cramers_v", y="feature", orientation="h",
                             color="Significant (p<0.05)", color_discrete_map={"Yes": "#2A9D8F", "No": "#B0B0B0"},
                             title="Association with severity (Cramér's V)")
                plot(fig, "ov_cramer")
                st.caption(f"The strongest factor has Cramér's V = {saved['feature_assoc']['cramers_v'].max():.2f} "
                           "(0 = no link, 1 = perfect link), so no single factor explains severity by itself.")
            if "severity_rate" in saved:
                sr = saved["severity_rate"]
                pick = pd.concat([sr.head(10), sr.tail(6)]).drop_duplicates().copy()
                pick["Category"] = pick["feature"].str.replace("_", " ") + " = " + pick["value"].astype(str)
                fig = px.bar(pick.sort_values("serious_or_fatal_%"), x="serious_or_fatal_%", y="Category",
                             orientation="h", color="serious_or_fatal_%", color_continuous_scale="OrRd",
                             title="Share of accidents that were serious or fatal (categories with ≥100 records)")
                plot(fig, "ov_sevrate")

# ---------------------------------------------------------------- FREQUENT PATTERNS
patterns, rules_all = None, None
mining_ready = len(mining_columns(base)) >= 3
if use_saved:
    if "patterns" in saved:
        patterns = saved["patterns"].copy()
        if ignore_common:
            one = patterns[patterns["size"] == 1]
            common = set(one.loc[(one["support"] > 0.80) & ~one["itemset"].str.startswith(SEV + "="), "itemset"])
            patterns = patterns[~patterns["itemset"].apply(lambda s: any(i in common for i in s.split(" AND ")))]
else:
    if mining_ready:
        freq, n_items = mine_freq(fdf, fkey, min_sup, max_len, ignore_common)
        patterns = pd.DataFrame(
            [(" AND ".join(sorted(k)), len(k), v, int(round(v * len(fdf)))) for k, v in freq.items()],
            columns=["itemset", "size", "support", "count"]).sort_values("support", ascending=False)
        rules_all = mine_rules(freq, fkey + f"|{min_sup}|{max_len}|{ignore_common}", min_conf, min_lift)

with tab_fp:
    st.header("🧩 Frequent patterns")
    explain("A <b>frequent pattern</b> is a group of conditions that appear together in many accidents. "
            "<b>Support</b> = the % of accidents that contain the whole pattern. "
            "Example: support 20% means 1 accident in 5 has all these conditions at once.")
    if not mining_ready:
        st.warning("⚠️ **Frequent pattern mining** is skipped because at least 3 usable categorical columns "
                   "(for example Light_conditions, Weather_conditions, Day_of_week) are required.")
    elif patterns is None or patterns.empty:
        st.info("No frequent patterns found with the current settings. Try lowering the minimum support.")
    else:
        if use_saved:
            st.caption("Showing the saved result of rta_data_mining.py (all accidents; the sidebar filters do not "
                       "change it). Choose 'Mine live' in the sidebar to mine the filtered accidents.")
        p1, p2 = st.columns(2)
        min_size = p1.slider("Minimum number of items in a pattern", 1, 4, 2, key="fp_size")
        top_n = p2.slider("Patterns to show in the chart", 5, 30, 15, key="fp_top")
        pt = patterns[patterns["size"] >= min_size].sort_values("support", ascending=False).reset_index(drop=True)
        if pt.empty:
            st.info("No pattern has that many items with the current settings.")
        else:
            top = pt.head(top_n).copy()
            top["Pattern"] = [f"{i + 1}. {short(t.replace(' AND ', ' + '), 85)}" for i, t in enumerate(top["itemset"])]
            top["Support (%)"] = top["support"] * 100
            top["Items in pattern"] = top["size"].astype(str)
            fig = px.bar(top.iloc[::-1], x="Support (%)", y="Pattern", orientation="h", color="Items in pattern",
                         color_discrete_sequence=VIVID, hover_data=["itemset", "count"],
                         title=f"Top {len(top)} most frequent patterns")
            plot(fig, "fp_chart")
            tbl = pt.head(200).copy()
            tbl["support"] = tbl["support"].round(4)
            show_df(tbl.rename(columns={"itemset": "pattern", "count": "accidents"}), height=320)
            if ignore_common:
                st.caption("Very common items (present in >80% of accidents) are ignored. "
                           "Untick the option in the sidebar to see the 'typical accident' profile.")

# ---------------------------------------------------------------- ASSOCIATION RULES
with tab_rules:
    st.header("🔗 Association rules (Apriori)")
    explain("A rule reads <b>IF antecedent THEN consequent</b>. "
            "<b>Support</b> = how many accidents contain both sides. "
            "<b>Confidence</b> = when the IF part is true, how often the THEN part is also true. "
            "<b>Lift</b> = how much more often the THEN part happens than by chance "
            "(lift 1 = no relation, above 1 = positive relation).")
    chosen = None
    if use_saved:
        avail = [k for k in RULE_SETS if k in saved]
        chosen = st.selectbox("Rule set", avail, format_func=lambda k: RULE_SETS[k], key="rule_set")
        rules_all = saved[chosen].copy()
        if chosen in ("rules_fatal", "rules_serious"):
            st.caption("These patterns were mined inside the fatal / serious group. Confidence = share of accidents "
                       "with the pattern that ended that way; 'cases_with_pattern' shows how many accidents support it "
                       "(small numbers = treat as a lead, not a proof).")
        st.caption("Saved result of rta_data_mining.py (all accidents; the sidebar filters do not change it).")
    elif not mining_ready:
        st.warning("⚠️ **Association rule mining** is skipped because at least 3 usable categorical columns "
                   "(for example Light_conditions, Weather_conditions, Day_of_week) are required.")
    else:
        st.caption(f"Rules mined live from the {len(fdf):,} selected accidents "
                   f"(support ≥ {min_sup:.0%}, confidence ≥ {min_conf:.0%}, lift ≥ {min_lift}). "
                   "Change the thresholds in the sidebar.")
        if has(SEV) and st.checkbox("Only rules that end in an accident-severity level", value=False, key="sev_only"):
            rules_all = rules_all[rules_all["consequent"].str.startswith(SEV + "=")]
            st.caption("Tip: Serious and Fatal accidents are rarer, so lower the minimum support to see their rules.")

    if rules_all is None or rules_all.empty:
        if mining_ready or use_saved:
            st.info("No rules found with these settings. Try lowering minimum support, confidence or lift.")
    else:
        r1, r2, r3 = st.columns([1, 1, 2])
        sort_by = r1.selectbox("Sort by", ["lift", "confidence", "support"], key="rule_sort")
        top_r = r2.slider("Rules in the chart", 5, 30, 12, key="rule_top")
        query = r3.text_input("Search in rules (e.g. Raining, Night, Pedestrian)", value="", key="rule_search")
        rules = rules_all.copy()
        if query.strip():
            q = query.strip().lower()
            rules = rules[(rules["antecedent"].str.lower().str.contains(q, regex=False))
                          | (rules["consequent"].str.lower().str.contains(q, regex=False))]
        rules = rules.sort_values(sort_by, ascending=False).reset_index(drop=True)
        st.write(f"**{len(rules):,}** rules shown")
        if rules.empty:
            st.info("No rule contains that text.")
        else:
            t = rules.iloc[0]
            st.success(f"**Top rule:** IF **{short(t['antecedent'], 200).replace(' AND ', ' + ')}** "
                       f"THEN **{short(t['consequent'], 100)}** - this happened in **{t['confidence']:.0%}** of the "
                       f"accidents where the IF part was true (confidence), covers **{t['support']:.1%}** of all "
                       f"accidents (support) and is **{t['lift']:.1f}×** more likely than by chance (lift).")

            main_cols = ["antecedent", "consequent", "support", "confidence", "lift"]
            extra = [c for c in rules.columns if c not in main_cols]
            disp = rules[main_cols + extra].copy()
            if "p_value" in disp.columns:
                disp["p_value"] = disp["p_value"].map(lambda v: f"{v:.1e}")
            cfg = {
                "support": st.column_config.NumberColumn("Support", format="%.3f"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                              format="%.2f"),
                "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
            }
            show_df(disp, column_config=cfg, height=340)
            st.download_button("⬇️ Download these rules (CSV)", rules.to_csv(index=False).encode("utf-8"),
                               "association_rules_view.csv", "text/csv", key="dl_rules")

            g1, g2 = st.columns(2)
            with g1:
                top = rules.head(top_r).copy()
                top["Rule"] = [f"{i + 1}. {short(a, 45)} → {short(c, 30)}"
                               for i, (a, c) in enumerate(zip(top["antecedent"], top["consequent"]))]
                fig = px.bar(top.iloc[::-1], x=sort_by, y="Rule", orientation="h", color="lift",
                             color_continuous_scale="Viridis", hover_data=["antecedent", "consequent", "support",
                                                                           "confidence", "lift"],
                             title=f"Top {len(top)} rules by {sort_by}")
                plot(fig, "rules_bar")
            with g2:
                fig = px.scatter(rules, x="support", y="confidence", color="lift", size="lift", size_max=16,
                                 color_continuous_scale="Plasma", hover_data=["antecedent", "consequent"],
                                 title="Support vs confidence (colour = lift)")
                plot(fig, "rules_scatter")

# ---------------------------------------------------------------- CLUSTERING
with tab_cl:
    st.header("🧬 Clustering - groups of similar accidents")
    explain("<b>K-Means</b> puts accidents into groups so that accidents in the same group are similar "
            "(driver age, vehicle type, junction, light, cause, time of day ...). "
            "The severity column is <b>not</b> used to build the groups - we only look at it afterwards to see "
            "whether the groups differ. The <b>PCA</b> map squeezes all features into 2 axes just to draw the groups.")
    Xc, cat_cols = get_Xc(base, data_key)
    if Xc is None:
        found = [c for c in CLUSTER_FEATURES if has(c)]
        st.warning("⚠️ **Clustering** is skipped because at least 3 of these columns are required: "
                   + ", ".join(f"`{c}`" for c in CLUSTER_FEATURES) + f". Found only {len(found)}.")
    else:
        saved_cols = [c for c in ["Cluster_best_silhouette", "Cluster_k4_finer_view"]
                      if mode == "project" and c in base.columns]
        choices = {}
        for c in saved_cols:
            kk = base[c].nunique()
            choices[f"Saved result: k = {kk} " + ("(best silhouette score)" if c.endswith("silhouette")
                                                   else "(finer view)")] = c
        choices["Run K-Means live (choose k)"] = None
        pick = st.radio("Clustering source", list(choices.keys()), horizontal=True, key="cl_src")
        if choices[pick] is not None:
            labels = base[choices[pick]]
            lkey = f"{data_key}|{choices[pick]}"
        else:
            k_live = st.slider("Number of clusters (k)", 2, 8, 4, key="cl_k")
            labels = pd.Series(run_kmeans(base, data_key, k_live), index=base.index)
            lkey = f"{data_key}|live|{k_live}"
        lab = labels.loc[fdf.index].map(lambda x: f"Cluster {x}")

        sil = silhouette_of(base, data_key, lkey, labels.values)
        m1, m2 = st.columns([1, 3])
        m1.metric("Silhouette score", f"{sil:.3f}")
        quality = ("weak - the groups overlap a lot" if sil < 0.25 else
                   "reasonable" if sil < 0.5 else "strong")
        m2.info(f"Silhouette measures how well separated the groups are (from -1 to 1). "
                f"Here the structure is **{quality}**. Low scores are common with categorical data, so read the "
                "clusters as tendencies, not as sharp boundaries.")

        cc1, cc2 = st.columns(2)
        with cc1:
            sizes = counts(lab, "Cluster").sort_values("Cluster")
            sizes["Share (%)"] = (sizes["Accidents"] / sizes["Accidents"].sum() * 100).round(1)
            fig = px.bar(sizes, x="Cluster", y="Accidents", color="Cluster", color_discrete_sequence=BOLD,
                         text="Share (%)", title="Cluster sizes (selected accidents)")
            fig.update_layout(showlegend=False)
            plot(fig, "cl_sizes")
        with cc2:
            pca_df, var = get_pca(base, data_key)
            pdf = pca_df.loc[fdf.index].copy()
            pdf["Cluster"] = lab
            fig = px.scatter(pdf, x="PC1", y="PC2", color="Cluster", opacity=0.6, color_discrete_sequence=BOLD,
                             title="PCA view of the clusters")
            plot(fig, "cl_pca")
            st.caption(f"The 2 PCA axes keep {var:.0%} of the information, so groups can look mixed on the map "
                       "even when they differ in the original features.")

        st.subheader("What describes each cluster?")
        prof = cluster_profile(Xc.loc[fdf.index, cat_cols], lab)
        if prof.empty:
            st.info("No category is clearly over-represented in these clusters (share ≥ 30% and lift ≥ 1.15).")
        else:
            st.caption("Lift = how much more common the category is inside the cluster than in all selected accidents.")
            prof = prof.copy()
            prof["Category"] = prof["category"].str.replace("_", " ")
            fig = px.bar(prof.sort_values("lift"), x="lift", y="Category", color="cluster", orientation="h",
                         color_discrete_sequence=BOLD, hover_data=["share_in_cluster", "share_overall"],
                         title="Characteristic categories per cluster")
            plot(fig, "cl_profile")
            show_df(prof[["cluster", "category", "share_in_cluster", "share_overall", "lift"]].round(3), height=260)

        if has(SEV):
            tmp = fdf[[SEV]].copy()
            tmp["Cluster"] = lab
            ct = tmp.dropna().groupby(["Cluster", SEV]).size().reset_index(name="Accidents")
            ct["Share (%)"] = ct["Accidents"] / ct.groupby("Cluster")["Accidents"].transform("sum") * 100
            fig = px.bar(ct, x="Cluster", y="Share (%)", color=SEV, barmode="stack",
                         color_discrete_map=sev_color_map(ct[SEV]), color_discrete_sequence=VIVID,
                         title="Severity inside each cluster (checked after clustering)")
            plot(fig, "cl_sev")
        else:
            st.warning(f"⚠️ **Severity by cluster** is skipped because the required column `{SEV}` is missing.")

        nums = [c for c in CLUSTER_NUMS if has(c)]
        if nums:
            tmp = fdf[nums].copy()
            tmp["Cluster"] = lab
            avg = tmp.groupby("Cluster")[nums].mean().round(2).reset_index()
            st.caption("Average number of vehicles / casualties per cluster")
            show_df(avg)

        if st.checkbox("Show silhouette score for k = 2 … 8 (takes a few seconds)", key="cl_sil_tab"):
            stbl = silhouette_table(base, data_key)
            fig = px.line(stbl, x="k", y="silhouette", markers=True, title="Silhouette score for different k")
            fig.update_xaxes(dtick=1)
            plot(fig, "cl_silplot")
        if choices[pick] is not None:
            key_prof = "profile_best" if choices[pick].endswith("silhouette") else "profile_k4"
            if key_prof in saved:
                with st.expander("Saved cluster profile from rta_results/ (all accidents)"):
                    show_df(saved[key_prof].round(3), height=260)

# ---------------------------------------------------------------- OUTLIERS
with tab_out:
    st.header("🚨 Outlier detection - unusual accidents")
    explain("An <b>outlier</b> is an accident that is very different from the others, for example a rare combination "
            "of road, weather, light and vehicle conditions. <b>Isolation Forest</b> finds such accidents by checking "
            "how easily each one can be separated from the rest. Outliers are <b>unusual</b>, not automatically "
            "more dangerous - the charts below show what makes them unusual.")
    if not mining_ready:
        st.warning("⚠️ **Outlier detection** is skipped because at least 3 usable categorical columns "
                   "(for example Light_conditions, Weather_conditions, Day_of_week) are required.")
    else:
        saved_out = mode == "project" and {"Outlier", "Anomaly_score"} <= set(base.columns)
        o_choices = []
        if saved_out:
            o_choices.append(f"Saved result ({base['Outlier'].mean():.1%} of accidents flagged)")
        o_choices.append("Run Isolation Forest live")
        o_pick = st.radio("Outlier source", o_choices, horizontal=True, key="out_src")
        if saved_out and o_pick == o_choices[0]:
            flag = base["Outlier"].astype(bool)
            score = base["Anomaly_score"]
        else:
            contam = st.slider("Share of accidents to flag as outliers (%)", 1, 10, 2, key="out_contam") / 100
            sc, fl = run_iforest(base, data_key, contam)
            flag, score = pd.Series(fl, index=base.index), pd.Series(sc, index=base.index)

        od = fdf.copy()
        od["Group"] = np.where(flag.loc[fdf.index].values, "Outlier", "Normal")
        od["Anomaly score"] = score.loc[fdf.index].values
        n_out = int((od["Group"] == "Outlier").sum())
        grp_colors = {"Normal": "#4C9AFF", "Outlier": "#FF4D4F"}

        o1, o2 = st.columns(2)
        with o1:
            gc = counts(od["Group"], "Group")
            fig = px.pie(gc, names="Group", values="Accidents", hole=0.5, color="Group",
                         color_discrete_map=grp_colors, title="Normal vs outlier accidents")
            plot(fig, "out_pie")
        with o2:
            nums = [c for c in CLUSTER_NUMS if has(c)]
            if nums:
                avg = od.groupby("Group")[nums].mean().reset_index().melt(id_vars="Group", var_name="Measure",
                                                                           value_name="Average")
                avg["Measure"] = avg["Measure"].str.replace("_", " ")
                fig = px.bar(avg, x="Measure", y="Average", color="Group", barmode="group",
                             color_discrete_map=grp_colors, title="Average vehicles and casualties")
                plot(fig, "out_avg")
            else:
                st.warning("⚠️ **Average vehicles / casualties comparison** is skipped because "
                           "`Number_of_vehicles_involved` and `Number_of_casualties` are missing.")

        o3, o4 = st.columns(2)
        with o3:
            if has(SEV) and n_out > 0:
                ct = od.dropna(subset=[SEV]).groupby(["Group", SEV]).size().reset_index(name="Accidents")
                ct["Share (%)"] = ct["Accidents"] / ct.groupby("Group")["Accidents"].transform("sum") * 100
                fig = px.bar(ct, x="Group", y="Share (%)", color=SEV, barmode="stack",
                             color_discrete_map=sev_color_map(ct[SEV]), color_discrete_sequence=VIVID,
                             title="Severity: normal vs outlier accidents")
                plot(fig, "out_sev")
            elif not has(SEV):
                st.warning(f"⚠️ **Severity comparison** is skipped because the required column `{SEV}` is missing.")
        with o4:
            pca_df, var = get_pca(base, data_key)
            if pca_df is not None:
                pdf = pca_df.loc[fdf.index].copy()
                pdf["Group"] = od["Group"].values
                pdf = pdf.sort_values("Group")
                fig = px.scatter(pdf, x="PC1", y="PC2", color="Group", opacity=0.65, color_discrete_map=grp_colors,
                                 title="Outliers on the PCA map")
                plot(fig, "out_pca")
            else:
                st.warning("⚠️ **PCA map of outliers** is skipped because clustering features are missing "
                           "(at least 3 of the clustering columns are needed).")

        st.subheader("What makes the outliers unusual?")
        if n_out < 5:
            st.info("Too few outliers in the current selection to describe them. Select more accidents or raise "
                    "the outlier share.")
        else:
            min_cnt = st.slider("Only show categories seen at least this many times among outliers",
                                3, 30, 10 if n_out >= 100 else 3, key="out_mincnt")
            ov = overrepresented(fdf, od["Group"].eq("Outlier"), mining_columns(base), min_cnt)
            if ov.empty:
                st.info("No category reaches that count. Lower the number above.")
            else:
                st.caption("Lift = how much more common the category is among outliers than among all selected "
                           "accidents (e.g. 5 = five times more common).")
                top = ov.head(12).copy()
                top["Category"] = top["category"].str.replace("_", " ").map(lambda s: short(s, 70))
                fig = px.bar(top.iloc[::-1], x="lift", y="Category", orientation="h", color="lift",
                             color_continuous_scale="Reds", hover_data=["share_in_outliers", "share_overall",
                                                                         "outlier_count"],
                             title="Categories over-represented among outliers")
                plot(fig, "out_chars")
                show_df(ov.head(40).round(3), height=260)
        if mode == "project" and "outlier_items" in saved:
            with st.expander("Saved outlier characteristics from rta_results/ (all accidents)"):
                show_df(saved["outlier_items"].rename(columns={"Unnamed: 0": "category"}).round(3), height=260)

        st.subheader("Most unusual accidents")
        show = [c for c in ["Day_of_week", "Time_period", "Type_of_vehicle", "Type_of_collision", "Cause_of_accident",
                            "Weather_conditions", "Light_conditions", "Number_of_vehicles_involved",
                            "Number_of_casualties", SEV] if c in od.columns]
        if show:
            tops = od.sort_values("Anomaly score", ascending=False).head(10)[["Anomaly score"] + show]
            show_df(tops.round(3))
        else:
            st.info("No descriptive columns are available to display the most unusual accidents.")

        st.subheader("Simple statistical outliers: number of casualties")
        if require(["Number_of_casualties"], "Casualty-count outliers (IQR rule)"):
            q1, q3 = base["Number_of_casualties"].quantile([0.25, 0.75])
            fence = q3 + 1.5 * (q3 - q1)
            n_iqr = int((fdf["Number_of_casualties"] > fence).sum())
            st.caption(f"IQR rule: accidents with more than **{fence:g}** casualties are statistical outliers "
                       f"({n_iqr:,} in the current selection).")
            hist = counts(fdf["Number_of_casualties"].dropna(), "Casualties").sort_values("Casualties")
            hist["Type"] = np.where(hist["Casualties"] > fence, "Above fence", "Normal range")
            fig = px.bar(hist, x="Casualties", y="Accidents", color="Type",
                         color_discrete_map={"Above fence": "#FF4D4F", "Normal range": "#4C9AFF"},
                         title="Casualties per accident")
            fig.update_xaxes(dtick=1)
            plot(fig, "out_iqr")

# ---------------------------------------------------------------- DATA & ABOUT
with tab_data:
    st.header("📁 Data & About")
    st.subheader("Which analyses are available for this dataset?")
    n_mine = len(mining_columns(base))
    n_clu = len([c for c in CLUSTER_FEATURES if has(c)])
    avail = [
        ("KPI cards", f"{SEV}, Number_of_casualties", has(SEV) and has("Number_of_casualties")),
        ("Severity distribution", SEV, has(SEV)),
        ("Accidents by hour", "Hour (or Time)", has("Hour")),
        ("Accidents by day", "Day_of_week", has("Day_of_week")),
        ("Top accident causes", "Cause_of_accident", has("Cause_of_accident")),
        ("Weather vs severity", f"Weather_conditions, {SEV}", has("Weather_conditions") and has(SEV)),
        ("Road condition vs severity", f"Road_surface_conditions, {SEV}",
         has("Road_surface_conditions") and has(SEV)),
        ("Frequent patterns & association rules", f"≥ 3 categorical columns (found {n_mine})", n_mine >= 3),
        ("Clustering + PCA", f"≥ 3 clustering columns (found {n_clu})", n_clu >= 3),
        ("Outlier detection", f"≥ 3 categorical columns (found {n_mine})", n_mine >= 3),
    ]
    show_df(pd.DataFrame({"Analysis": [a[0] for a in avail], "Needs": [a[1] for a in avail],
                          "Status": ["✅ available" if a[2] else "⚠️ missing columns" for a in avail]}))

    st.subheader("Expected columns found in this dataset")
    exp = EXPECTED_CAT + EXPECTED_NUM
    show_df(pd.DataFrame({"Expected column": exp, "Found": ["✅" if has(c) else "❌" for c in exp]}), height=260)

    st.subheader("Data preview (selected accidents)")
    prev_cols = [c for c in base.columns if c not in LABEL_COLS]
    show_df(fdf[prev_cols].head(100), height=300)
    st.download_button("⬇️ Download the selected accidents (CSV)", fdf[prev_cols].to_csv(index=False).encode("utf-8"),
                       "selected_accidents.csv", "text/csv", key="dl_data")

    st.subheader("About the methods")
    st.markdown(
        "- **Preprocessing** - unknown/NA values treated as missing, typos fixed, duplicates removed.\n"
        "- **Frequent patterns** - sets of conditions that occur together often (support).\n"
        "- **Association rules (Apriori)** - IF-THEN rules measured with support, confidence and lift.\n"
        "- **Clustering (K-Means)** - groups of similar accidents; quality checked with the silhouette score.\n"
        "- **Outlier detection (Isolation Forest + IQR)** - unusual accidents and what makes them unusual.\n"
        "- **No prediction** - the project describes patterns found in past data; it does not forecast accidents.\n"
        "- **Reading the results** - patterns show association, not proof of cause.")
