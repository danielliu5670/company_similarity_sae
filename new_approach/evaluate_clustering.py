#!/usr/bin/env python3
"""
Clustering evaluation: our approach vs. parent paper vs. SIC baseline.
Replicates the parent paper's MST + theta-sweep pipeline and evaluates under
both their unweighted MC(Gk) metric (each cluster weighted equally) and a
pair-weighted variant (each intra-cluster pair weighted equally).

Usage (Colab):
    !pip install tabulate
    !python evaluate_clustering.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl \
        --load-model /content/drive/MyDrive/company_similarity_sae/data/llama_selection_model.pkl \
        --top-k 1250
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tqdm import tqdm
import gc
import time


def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


# ================================================================
# Clustering utilities
# ================================================================

def build_msts(pairs_df, years, distance_col):
    """Build one MST per year from pairwise distances."""
    msts = {}
    for year in tqdm(years, desc="Building MSTs"):
        ydf = pairs_df[pairs_df["year"] == year]
        if ydf.empty:
            msts[year] = None
            continue
        G = nx.Graph()
        G.add_weighted_edges_from(zip(
            ydf["Company1"].values,
            ydf["Company2"].values,
            ydf[distance_col].values.astype(float),
        ))
        if G.number_of_edges() == 0:
            msts[year] = None
            continue
        msts[year] = nx.minimum_spanning_tree(G, weight="weight")
    return msts


def clusters_at_theta(mst, theta):
    """Cut MST edges above theta, return {id: [companies]}."""
    if mst is None:
        return {}
    g = mst.copy()
    g.remove_edges_from([
        (u, v) for u, v, d in g.edges(data=True) if d["weight"] > theta
    ])
    return {
        i: sorted(list(c))
        for i, c in enumerate(nx.connected_components(g), 1)
    }


def build_pair_lookup(pairs_df, years):
    """Precompute {year: {frozenset: correlation}} for fast cluster eval."""
    lookup = {}
    for year in years:
        ydf = pairs_df[pairs_df["year"] == year]
        c1s = ydf["Company1"].values
        c2s = ydf["Company2"].values
        corrs = ydf["correlation"].values
        d = {}
        for i in range(len(c1s)):
            d[frozenset((c1s[i], c2s[i]))] = float(corrs[i])
        lookup[year] = d
    return lookup


def evaluate_year_clusters(ylookup, clusters):
    """
    Evaluate clusters for one year.
    Returns (mc_unweighted, mc_pair_weighted, n_multi_clusters,
             n_intra_pairs, cluster_sizes).
    """
    cluster_means = []
    cluster_sizes = []
    total_corr = 0.0
    total_pairs = 0

    for companies in clusters.values():
        if len(companies) <= 1:
            continue
        corrs = []
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                c = ylookup.get(frozenset((companies[i], companies[j])))
                if c is not None:
                    corrs.append(c)
        if corrs:
            cluster_means.append(np.mean(corrs))
            cluster_sizes.append(len(companies))
            total_corr += sum(corrs)
            total_pairs += len(corrs)

    mc_uw = np.mean(cluster_means) if cluster_means else np.nan
    mc_pw = (total_corr / total_pairs) if total_pairs > 0 else np.nan
    return mc_uw, mc_pw, len(cluster_means), total_pairs, cluster_sizes


def evaluate_clusters(pair_lookup, year_clusters):
    """Evaluate over multiple years. Returns (mc_uw, mc_pw, details)."""
    uw_vals, pw_vals = [], []
    details = []

    for year, clusters in year_clusters:
        ylookup = pair_lookup.get(year, {})
        mc_uw, mc_pw, n_cl, n_pairs, sizes = evaluate_year_clusters(
            ylookup, clusters
        )
        uw_vals.append(mc_uw)
        pw_vals.append(mc_pw)
        details.append({
            "year": year,
            "n_clusters": n_cl,
            "n_intra_pairs": n_pairs,
            "mc_unweighted": mc_uw,
            "mc_pair_weighted": mc_pw,
            "median_size": int(np.median(sizes)) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "n_covered": sum(sizes),
        })

    valid_uw = [x for x in uw_vals if not np.isnan(x)]
    valid_pw = [x for x in pw_vals if not np.isnan(x)]
    return (
        np.mean(valid_uw) if valid_uw else np.nan,
        np.mean(valid_pw) if valid_pw else np.nan,
        details,
    )


def sweep_theta(msts, pair_lookup, years, thetas):
    """
    Sweep theta on given years, optimizing unweighted MC(Gk).
    Returns (best_theta, sweep_results).
    """
    best_theta = None
    best_mc = -np.inf
    results = []

    for theta in tqdm(thetas, desc="Theta sweep"):
        yc = []
        for year in years:
            if msts.get(year) is not None:
                yc.append((year, clusters_at_theta(msts[year], theta)))
        mc_uw, mc_pw, _ = evaluate_clusters(pair_lookup, yc)
        results.append({
            "theta": theta,
            "mc_unweighted": mc_uw,
            "mc_pair_weighted": mc_pw,
        })
        if not np.isnan(mc_uw) and mc_uw > best_mc:
            best_mc = mc_uw
            best_theta = theta

    return best_theta, results


# ================================================================
# Main
# ================================================================

P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--top-k", type=int, default=1250)
P.add_argument("--score-weight", action="store_true", default=True)
P.add_argument("--no-score-weight", action="store_false", dest="score_weight")
P.add_argument("--norm-alpha", type=float, default=0.0)
P.add_argument("--theta-min", type=float, default=-4.5)
P.add_argument("--theta-max", type=float, default=-1.0)
P.add_argument("--theta-step", type=float, default=0.1)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--original-pairs-ds", default="v1ctor10/cos_sim_4000pca_exp")
args = P.parse_args()


# ---- Load shared data ----
print("Loading pairs dataset...")
raw = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = raw.dropna(subset=["correlation", "cosine_similarity"]).copy()
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)
del raw; gc.collect()

all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
train_years = all_years[:split_idx]
test_years = all_years[split_idx:]
pop_corr = pairs_df["correlation"].mean()

print(f"  {len(pairs_df):,d} pairs, {all_years[0]}-{all_years[-1]}")
print(f"  Train: {train_years[0]}-{train_years[-1]} ({len(train_years)} yrs)")
print(f"  Test:  {test_years[0]}-{test_years[-1]} ({len(test_years)} yrs)")
print(f"  Population mean corr: {pop_corr:.4f}")

print("\nLoading company metadata...")
df_c = load_dataset(args.cov_ds)["train"].to_pandas()
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)


# ---- Load features + model ----
print("\nLoading features...")
df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)
del df_f; gc.collect()

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

saved = joblib.load(args.load_model)
scores = saved["scores"]
ranked = np.argsort(scores)[::-1]
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)
print(f"  {n_selected} features selected")

selected_features = feat_matrix[:, selected].copy().astype(np.float32)
if args.score_weight:
    weights = scores[selected].astype(np.float32)
    selected_features *= weights[np.newaxis, :]
norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10).astype(np.float32)
del feat_matrix; gc.collect()

# Map companies to feature indices
company_ids = df["__index_level_0__"].values
feat_idx_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})


# ---- Merge to common pair set ----
print("\nMerging pairs with feature indices...")
pairs_merged = pairs_df.merge(
    feat_idx_df.rename(columns={
        "__index_level_0__": "Company1", "feat_idx": "idx1",
    }),
    on=["Company1", "year"], how="inner",
)
pairs_merged = pairs_merged.merge(
    feat_idx_df.rename(columns={
        "__index_level_0__": "Company2", "feat_idx": "idx2",
    }),
    on=["Company2", "year"], how="inner",
)
print(f"  {len(pairs_merged):,d} common pairs")


# ---- Compute our similarities ----
print("Computing our similarities...")
idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
alpha = args.norm_alpha

sims_ours = np.empty(len(pairs_merged), dtype=np.float32)
batch = 500_000
for s in range(0, len(pairs_merged), batch):
    e = min(s + batch, len(pairs_merged))
    i1, i2 = idx1[s:e], idx2[s:e]
    dot = (selected_features[i1] * selected_features[i2]).sum(1)
    if alpha > 0:
        sims_ours[s:e] = dot / ((norms[i1] * norms[i2]) ** alpha)
    else:
        sims_ours[s:e] = dot

pairs_merged["sim_ours"] = sims_ours
del selected_features; gc.collect()


# ---- Convert to distances and standardize ----
pairs_merged["dist_pca"] = (
    1.0 - pairs_merged["cosine_similarity"].values
).astype(np.float32)
pairs_merged["dist_ours"] = (-sims_ours).astype(np.float32)

train_mask = pairs_merged["year"].isin(train_years)

scaler_A = StandardScaler()
scaler_A.fit(pairs_merged.loc[train_mask, ["dist_pca"]])
pairs_merged["dist_pca_s"] = scaler_A.transform(
    pairs_merged[["dist_pca"]]
).astype(np.float32).ravel()

scaler_B = StandardScaler()
scaler_B.fit(pairs_merged.loc[train_mask, ["dist_ours"]])
pairs_merged["dist_ours_s"] = scaler_B.transform(
    pairs_merged[["dist_ours"]]
).astype(np.float32).ravel()


# ---- Precompute pair lookup ----
print("\nBuilding pair lookup...")
t0 = time.time()
pair_lookup = build_pair_lookup(pairs_merged, all_years)
print(f"  Done in {time.time() - t0:.1f}s")

thetas = np.arange(
    args.theta_min, args.theta_max + args.theta_step / 2, args.theta_step
)
thetas = np.round(thetas, 2)


# ================================================================
# METHOD A: Parent paper (PCA cosine similarity)
# ================================================================
print("\n" + "=" * 70)
print("METHOD A: Parent paper (PCA 4000-dim cosine similarity)")
print("=" * 70)

msts_A = build_msts(pairs_merged, all_years, "dist_pca_s")
best_theta_A, sweep_A = sweep_theta(
    msts_A, pair_lookup, train_years, thetas
)

if best_theta_A is None:
    print("  WARNING: no valid theta found for parent paper.")
    best_theta_A = thetas[len(thetas) // 2]
elif best_theta_A == thetas[0] or best_theta_A == thetas[-1]:
    print(f"  WARNING: best theta at boundary ({best_theta_A:.2f}), "
          "consider expanding range.")

print(f"\n  Best theta (train, unweighted MC): {best_theta_A:.2f}")

test_cl_A = [
    (y, clusters_at_theta(msts_A[y], best_theta_A))
    for y in test_years if msts_A.get(y) is not None
]
mc_uw_A_oos, mc_pw_A_oos, det_A_oos = evaluate_clusters(
    pair_lookup, test_cl_A
)

all_cl_A = [
    (y, clusters_at_theta(msts_A[y], best_theta_A))
    for y in all_years if msts_A.get(y) is not None
]
mc_uw_A_all, mc_pw_A_all, _ = evaluate_clusters(pair_lookup, all_cl_A)

print(f"  OOS unweighted MC:    {mc_uw_A_oos:.4f}")
print(f"  OOS pair-weighted MC: {mc_pw_A_oos:.4f}")
del msts_A; gc.collect()


# ================================================================
# METHOD B: Our approach (supervised SAE)
# ================================================================
print("\n" + "=" * 70)
print(f"METHOD B: Our approach (k={n_selected}, "
      f"\u03b1={args.norm_alpha})")
print("=" * 70)

msts_B = build_msts(pairs_merged, all_years, "dist_ours_s")
best_theta_B, sweep_B = sweep_theta(
    msts_B, pair_lookup, train_years, thetas
)

if best_theta_B is None:
    print("  WARNING: no valid theta found for our approach.")
    best_theta_B = thetas[len(thetas) // 2]
elif best_theta_B == thetas[0] or best_theta_B == thetas[-1]:
    print(f"  WARNING: best theta at boundary ({best_theta_B:.2f}), "
          "consider expanding range.")

print(f"\n  Best theta (train, unweighted MC): {best_theta_B:.2f}")

test_cl_B = [
    (y, clusters_at_theta(msts_B[y], best_theta_B))
    for y in test_years if msts_B.get(y) is not None
]
mc_uw_B_oos, mc_pw_B_oos, det_B_oos = evaluate_clusters(
    pair_lookup, test_cl_B
)

all_cl_B = [
    (y, clusters_at_theta(msts_B[y], best_theta_B))
    for y in all_years if msts_B.get(y) is not None
]
mc_uw_B_all, mc_pw_B_all, _ = evaluate_clusters(pair_lookup, all_cl_B)

print(f"  OOS unweighted MC:    {mc_uw_B_oos:.4f}")
print(f"  OOS pair-weighted MC: {mc_pw_B_oos:.4f}")
del msts_B; gc.collect()


# ================================================================
# METHOD C: SIC code baseline
# ================================================================
print("\n" + "=" * 70)
print("METHOD C: SIC code baseline")
print("=" * 70)

sic_info = df_c[["__index_level_0__", "year", "sic_code"]].dropna(
    subset=["sic_code"]
).copy()
sic_info["year"] = sic_info["year"].astype(int)
sic_dedup = sic_info.drop_duplicates(
    subset=["__index_level_0__", "year"], keep="last"
)

sic_year_clusters = []
for year in all_years:
    yd = sic_dedup[sic_dedup["year"] == year]
    groups = yd.groupby("sic_code")["__index_level_0__"].apply(
        sorted
    ).to_dict()
    clusters = {i: v for i, (_, v) in enumerate(groups.items(), 1)}
    sic_year_clusters.append((year, clusters))

sic_test = [(y, c) for y, c in sic_year_clusters if y in test_years]
mc_uw_C_oos, mc_pw_C_oos, det_C_oos = evaluate_clusters(
    pair_lookup, sic_test
)
mc_uw_C_all, mc_pw_C_all, _ = evaluate_clusters(
    pair_lookup, sic_year_clusters
)

print(f"  OOS unweighted MC:    {mc_uw_C_oos:.4f}")
print(f"  OOS pair-weighted MC: {mc_pw_C_oos:.4f}")


# ================================================================
# Report
# ================================================================
yr_lo, yr_hi = min(test_years), max(test_years)

print("\n" + "=" * 70)
print("CLUSTERING COMPARISON")
print("=" * 70)

# Table 1: Unweighted MC(Gk)
print(f"\nUnweighted MC(Gk): each cluster weighted equally")
uw_table = [
    [f"Our approach (k={n_selected}, \u03b1={args.norm_alpha})",
     f"{mc_uw_B_oos:.4f}", f"{mc_uw_B_all:.4f}"],
    ["Parent paper (PCA cosine)",
     f"{mc_uw_A_oos:.4f}", f"{mc_uw_A_all:.4f}"],
    ["SIC code clusters",
     f"{mc_uw_C_oos:.4f}", f"{mc_uw_C_all:.4f}"],
    ["Population mean",
     f"{pop_corr:.4f}", f"{pop_corr:.4f}"],
]
print(tabulate(
    uw_table,
    headers=["Approach", f"OOS {yr_lo}-{yr_hi}", "All years"],
    tablefmt="simple_outline",
))

# Table 2: Pair-weighted MC(Gk)
print(f"\nPair-weighted MC(Gk): each intra-cluster pair weighted equally")
pw_table = [
    [f"Our approach (k={n_selected}, \u03b1={args.norm_alpha})",
     f"{mc_pw_B_oos:.4f}", f"{mc_pw_B_all:.4f}"],
    ["Parent paper (PCA cosine)",
     f"{mc_pw_A_oos:.4f}", f"{mc_pw_A_all:.4f}"],
    ["SIC code clusters",
     f"{mc_pw_C_oos:.4f}", f"{mc_pw_C_all:.4f}"],
    ["Population mean",
     f"{pop_corr:.4f}", f"{pop_corr:.4f}"],
]
print(tabulate(
    pw_table,
    headers=["Approach", f"OOS {yr_lo}-{yr_hi}", "All years"],
    tablefmt="simple_outline",
))


# Per-year breakdowns
def print_yearly(label, theta_val, details):
    print(f"\n{label} (\u03b8={theta_val:.2f}):")
    hdr = (f"  {'Year':>6s}  {'Clust':>6s}  {'Pairs':>10s}  "
           f"{'MC unwtd':>10s}  {'MC pwtd':>10s}  "
           f"{'Med sz':>8s}  {'Max sz':>8s}  {'Covered':>8s}")
    print(hdr)
    for d in sorted(details, key=lambda x: x["year"]):
        print(
            f"  {d['year']:>6d}  {d['n_clusters']:>6d}  "
            f"{d['n_intra_pairs']:>10,d}  "
            f"{d['mc_unweighted']:>10.4f}  "
            f"{d['mc_pair_weighted']:>10.4f}  "
            f"{d['median_size']:>8d}  {d['max_size']:>8d}  "
            f"{d['n_covered']:>8d}"
        )


print_yearly("Our approach OOS", best_theta_B, det_B_oos)
print_yearly("Parent paper OOS", best_theta_A, det_A_oos)
print_yearly("SIC baseline OOS", 0.0, det_C_oos)


# Theta sweep summaries
print(f"\nTheta sweep (train years):")

print(f"\n  Parent paper:")
print(f"  {'\u03b8':>8s}  {'MC unwtd':>10s}  {'MC pwtd':>10s}")
for r in sweep_A:
    m = " <--" if r["theta"] == best_theta_A else ""
    print(f"  {r['theta']:>8.2f}  {r['mc_unweighted']:>10.4f}  "
          f"{r['mc_pair_weighted']:>10.4f}{m}")

print(f"\n  Our approach:")
print(f"  {'\u03b8':>8s}  {'MC unwtd':>10s}  {'MC pwtd':>10s}")
for r in sweep_B:
    m = " <--" if r["theta"] == best_theta_B else ""
    print(f"  {r['theta']:>8.2f}  {r['mc_unweighted']:>10.4f}  "
          f"{r['mc_pair_weighted']:>10.4f}{m}")


print(f"\n{'=' * 70}")
print(f"Population mean return correlation: {pop_corr:.4f}")
print(f"Best \u03b8: parent paper = {best_theta_A:.2f}, "
      f"our approach = {best_theta_B:.2f}")
print(f"{'=' * 70}")