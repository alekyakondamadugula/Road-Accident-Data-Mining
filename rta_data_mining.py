import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save figures to files (works without a display)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 110)
sns.set_theme(style="whitegrid")

DATA_PATH = "RTA Dataset.csv"
OUT = "rta_results"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def show(table, n=15):
    """Print the first n rows of a result table (p-values in scientific notation)."""
    t = table.head(n).copy()
    for c in t.columns:
        if c == "p_value":
            t[c] = t[c].map(lambda v: f"{v:.1e}")
        elif t[c].dtype == float:
            t[c] = t[c].round(3)
    print(t.to_string(index=False))


def section(title):
    print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def savefig(name):
    for ax in plt.gcf().axes:                       # remove the stray "None" axis labels seaborn adds for Series input
        if ax.get_ylabel() == "None":
            ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


# =====================================================================================
# 1. LOAD AND INSPECT
# =====================================================================================
section("1. LOAD AND INSPECT")
raw = pd.read_csv(DATA_PATH)
print("Shape (rows, columns):", raw.shape)
print("\nData types:\n", raw.dtypes.value_counts())
print("\nFirst 3 rows:\n", raw.head(3).T.head(12))
print("\nExact duplicate rows:", raw.duplicated().sum())

miss = raw.isna().sum()
print("\nMissing values (NaN) per column:\n", miss[miss > 0].sort_values(ascending=False))

print("\nHidden 'unknown' style values (not NaN, but unusable):")
for c in raw.select_dtypes("object").columns:
    n = raw[c].astype(str).str.strip().str.lower().isin(["na", "unknown", "unknown or other"]).sum()
    if n:
        print(f"  {c:28s} {n}")

# =====================================================================================
# 2. PREPROCESSING
# =====================================================================================
section("2. PREPROCESSING")
df = raw.copy()

# 2a. Inconsistent text: stray spaces, typos, different spellings
for c in df.select_dtypes("object").columns:
    df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)
df["Pedestrian_movement"] = df["Pedestrian_movement"].str.replace("statioNot a Pedestrianry", "stationary", regex=False)
df["Fitness_of_casuality"] = df["Fitness_of_casuality"].replace({"NormalNormal": "Normal"})
df = df.replace({"other": "Other"})

# 2b. Missing / unknown / invalid codes  ->  NaN
df = df.replace({"na": np.nan, "Unknown": np.nan, "unknown": np.nan, "Unknown or other": np.nan})
df.loc[df["Defect_of_vehicle"].isin(["5", "7"]), "Defect_of_vehicle"] = np.nan          # undefined codes
df.loc[df["Age_band_of_casualty"] == "5", "Age_band_of_casualty"] = np.nan             # not an age band
df["Casualty_severity"] = df["Casualty_severity"].map({"1": "Level 1", "2": "Level 2", "3": "Level 3"})

# 2c. Duplicates
n_dup = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed after cleaning: {n_dup}")

# 2d. New useful columns from Time and numeric columns
t = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
df["Hour"] = t.dt.hour
df["Time_period"] = pd.cut(df["Hour"], bins=[-1, 5, 11, 17, 23],
                           labels=["Night (00-05)", "Morning (06-11)", "Afternoon (12-17)", "Evening (18-23)"]).astype(object)
df["Vehicles_involved"] = df["Number_of_vehicles_involved"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})
df["Casualties"] = df["Number_of_casualties"].clip(upper=3).astype(int).astype(str).replace({"3": "3+"})

# 2e. Drop columns that carry almost no usable information
def is_text(col):
    return not pd.api.types.is_numeric_dtype(df[col])      # works for object and newer "str" dtypes


cat_cols_all = [c for c in df.columns if is_text(c) and c != "Time"]
drop_reason = {}
for c in cat_cols_all:
    pct_missing = df[c].isna().mean()
    top_share = df[c].value_counts(normalize=True).iloc[0] if df[c].notna().any() else 1
    if pct_missing > 0.40:
        drop_reason[c] = f"{pct_missing:.0%} missing"
    elif top_share > 0.95:
        drop_reason[c] = f"one value covers {top_share:.0%} of known records"
print("\nColumns dropped from the analysis:")
for c, r in drop_reason.items():
    print(f"  {c:28s} -> {r}")
df = df.drop(columns=list(drop_reason) + ["Time"])

# Missing values are NOT imputed for pattern mining: a missing value simply produces no item.
mining_cols = [c for c in df.columns if is_text(c)]
print("\nMissing % in the columns kept:\n", (df[mining_cols].isna().mean() * 100).round(1).loc[lambda s: s > 0].sort_values(ascending=False))
print("\nFinal cleaned shape:", df.shape)
df.to_csv(os.path.join(OUT, "rta_cleaned.csv"), index=False)

# =====================================================================================
# 3. EXPLORATORY ANALYSIS
# =====================================================================================
section("3. EXPLORATORY DATA ANALYSIS")
sev_order = ["Slight Injury", "Serious Injury", "Fatal injury"]
sev_counts = df["Accident_severity"].value_counts().reindex(sev_order)
print("Accident severity:\n", pd.DataFrame({"count": sev_counts, "%": (sev_counts / len(df) * 100).round(2)}))
print("\nTop 10 causes:\n", df["Cause_of_accident"].value_counts().head(10))
print("\nAccidents by day:\n", df["Day_of_week"].value_counts().reindex(DAY_ORDER))
print("\nAccidents by time period:\n", df["Time_period"].value_counts())
print("\nNumeric summary:\n", df[["Number_of_vehicles_involved", "Number_of_casualties", "Hour"]].describe().round(2))

# Share of Serious + Fatal accidents for every category (categories with >= 100 records)
df["Serious_or_fatal"] = df["Accident_severity"].isin(["Serious Injury", "Fatal injury"]).astype(int)
overall_rate = df["Serious_or_fatal"].mean()
rows = []
for c in mining_cols:
    if c in ["Accident_severity"]:
        continue
    g = df.groupby(c)["Serious_or_fatal"].agg(["mean", "size"])
    for val, r in g.iterrows():
        if r["size"] >= 100:
            rows.append((c, val, int(r["size"]), r["mean"] * 100))
rate_tbl = pd.DataFrame(rows, columns=["feature", "value", "records", "serious_or_fatal_%"])
rate_tbl = rate_tbl.sort_values("serious_or_fatal_%", ascending=False)
rate_tbl.to_csv(os.path.join(OUT, "severity_rate_by_category.csv"), index=False)
print(f"\nOverall Serious+Fatal rate: {overall_rate*100:.1f}%")
print("Highest Serious+Fatal rate categories:\n", rate_tbl.head(10).round(1).to_string(index=False))
print("Lowest Serious+Fatal rate categories:\n", rate_tbl.tail(8).round(1).to_string(index=False))

# Association strength of each feature with severity (Cramer's V from chi-square test)
cv_rows = []
for c in mining_cols:
    if c == "Accident_severity":
        continue
    sub = df[[c, "Accident_severity"]].dropna()
    ct = pd.crosstab(sub[c], sub["Accident_severity"])
    chi2, p, _, _ = chi2_contingency(ct)
    v = np.sqrt(chi2 / (len(sub) * (min(ct.shape) - 1)))
    cv_rows.append((c, v, p))
cv = pd.DataFrame(cv_rows, columns=["feature", "cramers_v", "p_value"]).sort_values("cramers_v", ascending=False)
cv.to_csv(os.path.join(OUT, "feature_association_with_severity.csv"), index=False)
print("\nFeatures most associated with Accident_severity (Cramer's V; p>=0.05 means NOT significant):\n", cv.head(12).round(4).to_string(index=False))

# --- plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=sev_counts.index, y=sev_counts.values, ax=ax[0], palette="Reds")
ax[0].set_title("Accident severity"); ax[0].set_ylabel("Accidents"); ax[0].set_xlabel("")
sns.countplot(x="Day_of_week", data=df, order=DAY_ORDER, ax=ax[1], color="steelblue")
ax[1].set_title("Accidents by day of week"); ax[1].tick_params(axis="x", rotation=30); ax[1].set_xlabel("")
savefig("01_severity_and_day.png")

fig, ax = plt.subplots(figsize=(10, 4))
sns.countplot(x="Hour", data=df, color="steelblue", ax=ax)
ax.set_title("Accidents by hour of day")
savefig("02_accidents_by_hour.png")

fig, ax = plt.subplots(figsize=(9, 6))
top_c = df["Cause_of_accident"].value_counts().head(12)
sns.barplot(y=top_c.index, x=top_c.values, color="indianred", ax=ax)
ax.set_title("Top 12 causes of accidents"); ax.set_xlabel("Accidents")
savefig("03_top_causes.png")

fig, ax = plt.subplots(figsize=(10, 6))
pick = pd.concat([rate_tbl.head(10), rate_tbl.tail(6)])
sns.barplot(y=pick["feature"] + " = " + pick["value"].str[:35], x=pick["serious_or_fatal_%"],
            color="darkorange", ax=ax)
ax.axvline(overall_rate * 100, color="black", ls="--", label=f"overall {overall_rate*100:.1f}%")
ax.set_title("Serious + Fatal rate by category (>= 100 records)", loc="right")
ax.set_xlabel("% of accidents that were Serious or Fatal"); ax.legend()
savefig("04_severity_rate_by_category.png")

fig, ax = plt.subplots(figsize=(8, 6))
cv_sig = cv[cv["p_value"] < 0.05].head(12)
sns.barplot(y=cv_sig["feature"], x=cv_sig["cramers_v"], color="teal", ax=ax)
ax.set_title("Features significantly associated with severity (p<0.05, Cramer's V)")
savefig("05_feature_association_with_severity.png")

hm = df.pivot_table(index="Day_of_week", columns="Time_period", values="Hour", aggfunc="size").reindex(DAY_ORDER)
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(hm, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
ax.set_title("Accident count: day vs time period"); ax.set_xlabel(""); ax.set_ylabel("")
savefig("06_day_vs_period_heatmap.png")

# =====================================================================================
# 4. FREQUENT PATTERN MINING  +  5. ASSOCIATION RULES (APRIORI)
# =====================================================================================
section("4-5. FREQUENT PATTERNS AND ASSOCIATION RULES (APRIORI)")

# Each accident = one transaction; each "column=value" is an item. Missing values give no item.
item_cols = [c for c in mining_cols if c != "Serious_or_fatal"]
onehot = pd.get_dummies(df[item_cols], prefix_sep="=", dtype=bool)
sev_items = [c for c in onehot.columns if c.startswith("Accident_severity=")]
print(f"Transactions: {len(onehot)}   Distinct items: {onehot.shape[1]}")


def apriori(data, min_support, max_len=4):
    """Level-wise Apriori. Returns {frozenset(items): support}."""
    cols = {c: data[c].values for c in data.columns}
    freq, current = {}, []
    for c, arr in cols.items():                         # 1-itemsets
        s = arr.mean()
        if s >= min_support:
            freq[frozenset([c])] = s
            current.append((c,))
    current.sort()
    k = 2
    while current and k <= max_len:
        prev = set(current)
        by_prefix = {}
        for itemset in current:                         # join step (same prefix)
            by_prefix.setdefault(itemset[:-1], []).append(itemset[-1])
        nxt = []
        for prefix, lasts in by_prefix.items():
            for a, b in combinations(sorted(lasts), 2):
                cand = prefix + (a, b)
                if k > 2 and not all(sub in prev for sub in combinations(cand, k - 1)):
                    continue                            # prune step (Apriori property)
                s = np.logical_and.reduce([cols[i] for i in cand]).mean()
                if s >= min_support:
                    freq[frozenset(cand)] = s
                    nxt.append(cand)
        current = sorted(nxt)
        k += 1
    return freq


def make_rules(freq, min_conf, min_lift, min_improvement=0.02):
    """Rules with ONE item in the consequent: support, confidence, lift.
    Redundant rules are removed: a rule A+B -> C is dropped when a simpler rule (a subset of its
    antecedent) already gives almost the same confidence (improvement < min_improvement)."""
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
            if len(ante) >= 2:                                   # redundancy check
                best_sub = max(freq[sub | {c}] / freq[sub]
                               for r in range(1, len(ante))
                               for sub in map(frozenset, combinations(ante, r)))
                if conf - best_sub < min_improvement:
                    continue
            rows.append((" AND ".join(sorted(ante)), c, sup, conf, lift, len(itemset)))
    out = pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift", "size"])
    return out.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# ---- 4. Frequent itemsets
MIN_SUP, MIN_CONF, MIN_LIFT = 0.05, 0.60, 1.20
freq_all = apriori(onehot, MIN_SUP, max_len=4)
fi = pd.DataFrame([(" AND ".join(sorted(k)), len(k), v, round(v * len(onehot))) for k, v in freq_all.items()],
                  columns=["itemset", "size", "support", "count"]).sort_values("support", ascending=False)
fi.to_csv(os.path.join(OUT, "frequent_itemsets.csv"), index=False)
print(f"Frequent itemsets (min_support={MIN_SUP}): {len(fi)}")
print(fi.groupby("size").size().rename("itemsets").to_string())
print("\nTop frequent itemsets with 2+ items:\n", fi[fi["size"] >= 2].head(15).round(3).to_string(index=False))
print("\nMost frequent 4-item patterns (= the 'typical accident' profile):\n", fi[fi["size"] == 4].head(5).round(3).to_string(index=False))

# For association RULES, items present in more than 80% of accidents (e.g. male driver, asphalt road) are left out:
# they appear everywhere, so they can never form an interesting rule (lift stays close to 1).
ubiquitous = onehot.columns[(onehot.mean() > 0.80) & ~onehot.columns.isin(sev_items)]
print("\nItems excluded from rule mining (support > 80%):", list(ubiquitous))
onehot_rules = onehot.drop(columns=ubiquitous)
freq = apriori(onehot_rules, MIN_SUP, max_len=4)

# ---- 5a. General association rules
rules = make_rules(freq, MIN_CONF, MIN_LIFT)
rules.to_csv(os.path.join(OUT, "association_rules_general.csv"), index=False)
print(f"\nGeneral rules (support>={MIN_SUP}, confidence>={MIN_CONF}, lift>={MIN_LIFT}): {len(rules)}")
show(rules, 20)

# ---- 5b. Rules whose consequent is an accident-severity level (support lowered for Serious injury)
freq_sev = apriori(onehot_rules, 0.02, max_len=3)
rules_sev_all = make_rules(freq_sev, 0.0, 1.0)
rules_sev = rules_sev_all[rules_sev_all["consequent"].isin(sev_items)]
rules_sev.to_csv(os.path.join(OUT, "association_rules_severity_support2pct.csv"), index=False)
print("\nRules ending in a severity level (min_support=0.02, lift>=1.0): %d" % len(rules_sev))
show(rules_sev, 10)


# ---- 5b-2. More specific rules found with a lower support threshold (2%)
rules_specific = make_rules(freq_sev, 0.50, 1.50)
rules_specific.to_csv(os.path.join(OUT, "association_rules_specific_support2pct.csv"), index=False)
print(f"\nSpecific rules (support>=0.02, confidence>=0.50, lift>=1.5): {len(rules_specific)}")
show(rules_specific, 20)


# ---- 5c. Targeted mining for the rare severity classes (Serious / Fatal)
# Fatal accidents are only ~1.3% of the data, so they never reach a global support threshold.
# Method: mine frequent itemsets INSIDE the severity class, then measure how often each pattern
# leads to that severity in the whole dataset (confidence) and compare with the base rate (lift).
def class_rules(sev_item, min_sub_support, min_cases=30, min_lift=1.5, max_len=3):
    mask_sev = onehot[sev_item].values
    sub = onehot.loc[mask_sev, [c for c in onehot.columns if c not in sev_items]]
    f = apriori(sub, min_sub_support, max_len)
    base = mask_sev.mean()
    stats = {}                                           # itemset -> (cases with pattern, cases that had this severity)
    for iset, s in f.items():
        n_sev = int(round(s * len(sub)))
        n_all = int(np.logical_and.reduce([onehot[i].values for i in iset]).sum())
        stats[iset] = (n_all, n_sev)
    rows = []
    for iset, (n_all, n_sev) in stats.items():
        if n_all < min_cases:
            continue
        conf = n_sev / n_all
        lift = conf / base
        if lift < min_lift:
            continue
        # keep only patterns clearly stronger than every simpler pattern contained in them
        subs = [stats[frozenset(x)] for r in range(1, len(iset)) for x in combinations(iset, r)]
        if subs and conf < 1.10 * max(b / a for a, b in subs):
            continue
        pval = binomtest(n_sev, n_all, base, alternative="greater").pvalue
        rows.append((" AND ".join(sorted(iset)), sev_item.split("=")[1], n_sev / len(onehot), conf, lift, n_all, n_sev, pval))
    out = pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift",
                                      "cases_with_pattern", "of_which_this_severity", "p_value"])
    return out.sort_values(["lift", "cases_with_pattern"], ascending=False).reset_index(drop=True)


fatal_rules = class_rules("Accident_severity=Fatal injury", 0.10)
serious_rules = class_rules("Accident_severity=Serious Injury", 0.15, min_cases=100, min_lift=1.2)
fatal_rules.to_csv(os.path.join(OUT, "association_rules_fatal.csv"), index=False)
serious_rules.to_csv(os.path.join(OUT, "association_rules_serious.csv"), index=False)
print(f"\nFatal-injury patterns (frequent in fatal accidents, >=30 cases overall, lift>=1.5, simpler-pattern pruning): {len(fatal_rules)}")
show(fatal_rules, 12)
print("\nSingle factors (one item) most linked to Fatal injury:")
show(fatal_rules[~fatal_rules["antecedent"].str.contains(" AND ")], 10)
print(f"\nSerious-injury patterns (>=100 cases overall, lift>=1.2): {len(serious_rules)}")
show(serious_rules, 12)

# ---- plots
fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", s=25, alpha=0.8)
plt.colorbar(sc, label="lift")
ax.set_xlabel("support"); ax.set_ylabel("confidence"); ax.set_title("General association rules (Apriori)")
savefig("07_rules_scatter.png")

fig, ax = plt.subplots(figsize=(11, 6))
top = rules.head(12).copy()
labels = [f"{a[:55]}  ->  {c[:35]}" for a, c in zip(top["antecedent"], top["consequent"])]
sns.barplot(y=labels, x=top["lift"], color="slateblue", ax=ax)
ax.set_title("Top 12 rules by lift"); ax.set_xlabel("lift")
savefig("08_top_rules_by_lift.png")

for name, tbl, fn in [("Fatal", fatal_rules, "09_fatal_patterns.png"), ("Serious", serious_rules, "10_serious_patterns.png")]:
    if len(tbl):
        top = tbl.head(12)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(y=[a[:90] for a in top["antecedent"]], x=top["lift"], color="firebrick", ax=ax)
        ax.set_title(f"Patterns most strongly linked to {name} accidents (lift)"); ax.set_xlabel("lift")
        savefig(fn)

# =====================================================================================
# 6. CLUSTERING (K-Means on one-hot + scaled numeric features)
# =====================================================================================
section("6. CLUSTERING - groups of similar accidents")
cluster_cats = [c for c in ["Age_band_of_driver", "Driving_experience", "Type_of_vehicle", "Area_accident_occured",
                            "Lanes_or_Medians", "Types_of_Junction", "Light_conditions", "Road_surface_conditions",
                            "Type_of_collision", "Vehicle_movement", "Cause_of_accident", "Time_period"] if c in df.columns]
cdf = df[cluster_cats].copy()
n_imputed = int(cdf.isna().sum().sum())
for c in cluster_cats:                                   # clustering needs complete data -> mode imputation
    cdf[c] = cdf[c].fillna(cdf[c].mode()[0])
    vc = cdf[c].value_counts(normalize=True)             # merge very rare categories (<2%)
    cdf[c] = cdf[c].where(~cdf[c].isin(vc[vc < 0.02].index), "Rare")
Xc = pd.get_dummies(cdf, prefix_sep="=", dtype=float)
for c in ["Number_of_vehicles_involved", "Number_of_casualties"]:
    v = df[c].clip(upper=4)
    Xc[c] = (v - v.min()) / (v.max() - v.min())          # min-max scale to 0-1 like the one-hot columns
print(f"Features used: {cluster_cats} + number of vehicles + number of casualties")
print(f"Missing cells filled with the column mode for clustering only: {n_imputed}")
print("Matrix shape:", Xc.shape)

rng = np.random.RandomState(SEED)
sample_idx = rng.choice(len(Xc), 4000, replace=False)
k_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xc)
    sil = silhouette_score(Xc.values[sample_idx], km.labels_[sample_idx])
    k_rows.append((k, km.inertia_, sil))
ktab = pd.DataFrame(k_rows, columns=["k", "inertia", "silhouette"])
print("\n", ktab.round(4).to_string(index=False))
best_k = int(ktab.loc[ktab["silhouette"].idxmax(), "k"])
print(f"\nBest k by silhouette score: {best_k}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ktab["k"], ktab["inertia"], "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
ax[1].plot(ktab["k"], ktab["silhouette"], "o-", color="green"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
savefig("11_choose_k.png")

pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(Xc)
print(f"\nPCA 2D explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance (used for plots only)")
cat_onehot = Xc.drop(columns=["Number_of_vehicles_involved", "Number_of_casualties"])
overall_share = cat_onehot.mean()


def profile_clusters(k, tag):
    """Fit K-Means with k clusters, print/save the profile of each cluster and draw the plots."""
    labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xc)
    df[f"Cluster_{tag}"] = labels
    profile_rows = []
    print(f"\n########## K-Means with k={k} ({tag}) ##########")
    print("Cluster sizes:", df[f"Cluster_{tag}"].value_counts().sort_index().to_dict())
    print("(categories with >=30% share inside the cluster and lift >= 1.15; lift = share in cluster / share overall)")
    for cl in range(k):
        m = labels == cl
        share = cat_onehot[m].mean()
        chars = pd.DataFrame({"share_in_cluster": share, "share_overall": overall_share, "lift": share / overall_share})
        chars = chars[(chars["share_in_cluster"] >= 0.30) & (chars["lift"] >= 1.15)].sort_values("lift", ascending=False)
        sev = df.loc[m, "Accident_severity"].value_counts(normalize=True).reindex(sev_order).fillna(0) * 100
        print(f"\n--- Cluster {cl}: {m.sum()} accidents ({m.mean()*100:.1f}%) | avg vehicles "
              f"{df.loc[m,'Number_of_vehicles_involved'].mean():.2f}, avg casualties {df.loc[m,'Number_of_casualties'].mean():.2f}")
        print(f"    Severity %: Slight {sev.iloc[0]:.1f} | Serious {sev.iloc[1]:.1f} | Fatal {sev.iloc[2]:.1f}")
        for item, r in chars.head(8).iterrows():
            print(f"    {item:62s} in cluster {r['share_in_cluster']*100:5.1f}%  overall {r['share_overall']*100:5.1f}%  lift {r['lift']:.2f}")
            profile_rows.append((cl, item, r["share_in_cluster"], r["share_overall"], r["lift"]))
    pd.DataFrame(profile_rows, columns=["cluster", "item", "share_in_cluster", "share_overall", "lift"]).to_csv(
        os.path.join(OUT, f"cluster_profiles_{tag}.csv"), index=False)
    sev_by_cl = pd.crosstab(df[f"Cluster_{tag}"], df["Accident_severity"], normalize="index")[sev_order] * 100
    print("\nSeverity % by cluster (severity was NOT used to form the clusters):\n", sev_by_cl.round(2))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette="tab10", s=10, alpha=0.6, ax=ax)
    ax.set_title(f"Accident clusters (K-Means, k={k}) in PCA space"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    savefig(f"12_clusters_pca_{tag}.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sev_by_cl.plot(kind="bar", stacked=True, ax=ax, color=["#6baed6", "#fd8d3c", "#a50f15"])
    ax.set_ylabel("% of accidents in cluster"); ax.set_title(f"Severity inside each cluster (k={k})"); ax.tick_params(axis="x", rotation=0)
    savefig(f"13_cluster_severity_{tag}.png")


profile_clusters(best_k, "best_silhouette")
if best_k != 4:
    profile_clusters(4, "k4_finer_view")
df["Cluster"] = df["Cluster_best_silhouette"]

# =====================================================================================
# 7. OUTLIER DETECTION
# =====================================================================================
section("7. OUTLIER DETECTION")
# (a) Statistical: IQR rule on number of casualties
q1, q3 = df["Number_of_casualties"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
iqr_out = df[df["Number_of_casualties"] > upper]
print(f"(a) IQR rule on Number_of_casualties: Q1={q1}, Q3={q3}, upper fence={upper} -> {len(iqr_out)} accidents above the fence")
print(iqr_out["Number_of_casualties"].value_counts().sort_index().to_string())

# (b) Multivariate: Isolation Forest on the same encoded features (finds unusual COMBINATIONS)
Xo = onehot.drop(columns=sev_items).astype(float)       # all categories (rare ones are NOT merged); severity left out
Xo["Number_of_vehicles_involved"] = (df["Number_of_vehicles_involved"].clip(upper=4) - 1) / 3
Xo["Number_of_casualties"] = (df["Number_of_casualties"].clip(upper=4) - 1) / 3
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=SEED).fit(Xo)
df["Anomaly_score"] = -iso.score_samples(Xo)            # higher = more unusual
df["Outlier"] = iso.predict(Xo) == -1
out = df[df["Outlier"]]
print(f"\n(b) Isolation Forest (contamination=2%): {len(out)} outlier accidents")
sev_o = out["Accident_severity"].value_counts(normalize=True).reindex(sev_order).fillna(0) * 100
sev_a = df["Accident_severity"].value_counts(normalize=True).reindex(sev_order) * 100
print("Severity % -> outliers vs all:\n", pd.DataFrame({"outliers": sev_o, "all": sev_a}).round(2))
print(f"Avg casualties: outliers {out['Number_of_casualties'].mean():.2f} vs all {df['Number_of_casualties'].mean():.2f}")
print(f"Avg vehicles  : outliers {out['Number_of_vehicles_involved'].mean():.2f} vs all {df['Number_of_vehicles_involved'].mean():.2f}")

onehot_all = onehot
share_out = onehot_all[df["Outlier"].values].mean()
share_all = onehot_all.mean()
ov = pd.DataFrame({"share_in_outliers": share_out, "share_overall": share_all, "lift": share_out / share_all,
                   "count_in_outliers": onehot_all[df["Outlier"].values].sum()})
ov = ov[ov["count_in_outliers"] >= 10].sort_values("lift", ascending=False)
ov.to_csv(os.path.join(OUT, "outlier_overrepresented_items.csv"))
print("\nWhat makes the outliers unusual (categories over-represented among outliers):\n", ov.head(12).round(3).to_string())

show_cols = ["Anomaly_score", "Day_of_week", "Time_period", "Type_of_vehicle", "Type_of_collision", "Cause_of_accident",
             "Number_of_vehicles_involved", "Number_of_casualties", "Accident_severity"]
top_out = df.sort_values("Anomaly_score", ascending=False).head(10)[show_cols]
print("\nTop 10 most unusual accidents:\n", top_out.round(3).to_string())
df[df["Outlier"]].sort_values("Anomaly_score", ascending=False).to_csv(os.path.join(OUT, "outliers.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Z[~df["Outlier"], 0], Z[~df["Outlier"], 1], s=8, alpha=0.3, label="normal")
ax.scatter(Z[df["Outlier"], 0], Z[df["Outlier"], 1], s=18, color="red", label="outlier")
ax.set_title("Isolation Forest outliers in PCA space"); ax.legend(); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
savefig("14_outliers_pca.png")

fig, ax = plt.subplots(figsize=(9, 5))
t10 = ov.head(12)
sns.barplot(y=t10.index.str[:60], x=t10["lift"], color="crimson", ax=ax)
ax.set_title("Categories over-represented among outlier accidents (lift)"); ax.set_xlabel("lift")
savefig("15_outlier_characteristics.png")

# =====================================================================================
# 8-10. SAVE RESULTS
# =====================================================================================
section("8-10. SAVED FILES")
df.to_csv(os.path.join(OUT, "rta_with_clusters_and_outliers.csv"), index=False)
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        print(os.path.join(root, f))
print("\nDone.")
