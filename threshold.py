#!/usr/bin/env python3
"""
Sweep MST cut-off threshold θ using a 75/25 temporal split.
- Fit θ on the first 75% of years (in-sample).
- Report final correlation on the remaining 25% (out-of-sample).
- Also reports K-means and SIC correlations at the best θ.

Usage (Colab):
    !python sweep_threshold.py \
        --pairs-pkl /content/drive/MyDrive/company_similarity_sae/data/gemma_pairs.pkl \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/gemma_features.pkl \
        --pca-model /content/drive/MyDrive/company_similarity_sae/data/gemma_pca_model.pkl
"""

import argparse
import os
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm


def perform_clustering_per_year(pairs_df, years, threshold):
    results = []
    for year in years:
        year_df = pairs_df[pairs_df["year"] == year]
        if year_df.empty:
            results.append({"year": year, "clusters": {}})
            continue

        G = nx.Graph()
        edges = list(
            zip(
                year_df["Company1"],
                year_df["Company2"],
                year_df["cosine_distance_scaled"],
            )
        )
        G.add_weighted_edges_from(edges)

        if G.number_of_edges() == 0:
            results.append({"year": year, "clusters": {}})
            continue

        mst = nx.minimum_spanning_tree(G, weight="weight")
        edges_to_remove = [
            (u, v) for u, v, d in mst.edges(data=True) if d["weight"] > threshold
        ]
        mst.remove_edges_from(edges_to_remove)

        clusters = list(nx.connected_components(mst))
        cluster_dict = {
            idx: sorted(list(c)) for idx, c in enumerate(clusters, start=1)
        }
        cluster_dict = {k: v for k, v in cluster_dict.items() if k != -1}

        results.append({"year": year, "clusters": cluster_dict})

    return pd.DataFrame(results)


def calculate_avg_correlation(pairs_df, cluster_df):
    avg_correlations = []
    for _, row in cluster_df.iterrows():
        year = row["year"]
        clusters = row["clusters"]
        year_data = pairs_df[pairs_df["year"] == year]
        cluster_stats = []

        for companies in clusters.values():
            if len(companies) <= 1:
                continue
            cluster_pairs = year_data[
                year_data["Company1"].isin(companies)
                & year_data["Company2"].isin(companies)
            ]
            if not cluster_pairs.empty:
                cluster_stats.append(cluster_pairs["correlation"].mean())

        if cluster_stats:
            avg_correlations.append({"year": year, "avg_corr": np.mean(cluster_stats)})
        else:
            avg_correlations.append({"year": year, "avg_corr": np.nan})

    return pd.DataFrame(avg_correlations)


def perform_kmeans_clustering_per_year(feat_df, years, target_k_per_year):
    results = []
    for year in years:
        year_data = feat_df[feat_df["year"] == year]
        if len(year_data) < 2:
            results.append({"year": year, "clusters": {}})
            continue

        k = target_k_per_year.get(year, 5)
        k = min(k, len(year_data))
        if k < 2:
            results.append({"year": year, "clusters": {}})
            continue

        Z_cpu = np.vstack(year_data["pca_features"].values)
        kmeans = KMeans(n_clusters=min(k, len(Z_cpu)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(Z_cpu)

        cluster_dict = {}
        ids = year_data["__index_level_0__"].values
        for label in np.unique(labels):
            companies = sorted(ids[labels == label].tolist())
            cluster_dict[int(label) + 1] = companies

        results.append({"year": year, "clusters": cluster_dict})

    return pd.DataFrame(results)


P = argparse.ArgumentParser()
P.add_argument("--pairs-pkl", required=True)
P.add_argument("--features-pkl", default=None)
P.add_argument("--pca-model", default=None)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--theta-min", type=float, default=-4.5)
P.add_argument("--theta-max", type=float, default=-1.0)
P.add_argument("--theta-step", type=float, default=0.1)
args = P.parse_args()

# ------------------------------------------------------------------
# Load pairs
# ------------------------------------------------------------------
print("Loading pairs...")
pairs_df = pd.read_pickle(args.pairs_pkl)
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["cosine_distance"] = 1 - pairs_df["cosine_similarity"]

all_years = sorted(pairs_df["year"].unique())
n_total = len(all_years)
split_idx = int(0.75 * n_total)
train_years = all_years[:split_idx]
test_years = all_years[split_idx:]

print(f"  Total years: {all_years[0]}-{all_years[-1]} ({n_total} years)")
print(f"  Train years: {train_years[0]}-{train_years[-1]} ({len(train_years)} years)")
print(f"  Test years:  {test_years[0]}-{test_years[-1]} ({len(test_years)} years)")

# Fit scaler on train years only
scaler = StandardScaler()
train_mask = pairs_df["year"].isin(train_years)
scaler.fit(pairs_df.loc[train_mask, ["cosine_distance"]])
pairs_df["cosine_distance_scaled"] = scaler.transform(pairs_df[["cosine_distance"]])

# ------------------------------------------------------------------
# Sweep θ on train years
# ------------------------------------------------------------------
thetas = np.arange(args.theta_min, args.theta_max + args.theta_step / 2, args.theta_step)
thetas = np.round(thetas, 2)

print(f"\nSweeping {len(thetas)} thresholds on train years...")
best_theta = None
best_train_corr = -np.inf
sweep_results = []

for theta in tqdm(thetas, desc="Theta sweep"):
    cluster_df = perform_clustering_per_year(pairs_df, train_years, theta)
    corr_df = calculate_avg_correlation(pairs_df, cluster_df)
    mean_corr = corr_df["avg_corr"].mean()
    sweep_results.append({"theta": theta, "train_mean_corr": mean_corr})

    if not np.isnan(mean_corr) and mean_corr > best_train_corr:
        best_train_corr = mean_corr
        best_theta = theta

print(f"\n  Best θ = {best_theta:.2f}  (train MC = {best_train_corr:.4f})")

# Print full sweep for reference
print("\n  Sweep summary:")
print(f"  {'theta':>8s}  {'train_MC':>10s}")
for r in sweep_results:
    marker = " <-- best" if r["theta"] == best_theta else ""
    print(f"  {r['theta']:>8.2f}  {r['train_mean_corr']:>10.4f}{marker}")

# ------------------------------------------------------------------
# Evaluate best θ on test years (out-of-sample)
# ------------------------------------------------------------------
print(f"\nEvaluating θ = {best_theta:.2f} on test years ({test_years[0]}-{test_years[-1]})...")
test_cluster_df = perform_clustering_per_year(pairs_df, test_years, best_theta)
test_corr_df = calculate_avg_correlation(pairs_df, test_cluster_df)
test_mean_corr = test_corr_df["avg_corr"].mean()

# Count MST clusters per year (for K-means matching)
mst_cluster_counts = {}
for _, row in test_cluster_df.iterrows():
    k = len([c for c in row["clusters"].values() if len(c) > 1])
    mst_cluster_counts[row["year"]] = max(k, 2)

# Also compute on all years for reference
all_cluster_df = perform_clustering_per_year(pairs_df, all_years, best_theta)
all_corr_df = calculate_avg_correlation(pairs_df, all_cluster_df)
all_mean_corr = all_corr_df["avg_corr"].mean()

all_mst_cluster_counts = {}
for _, row in all_cluster_df.iterrows():
    k = len([c for c in row["clusters"].values() if len(c) > 1])
    all_mst_cluster_counts[row["year"]] = max(k, 2)

# ------------------------------------------------------------------
# K-means (if features + PCA model provided)
# ------------------------------------------------------------------
kmeans_test_corr = np.nan
kmeans_all_corr = np.nan

if args.features_pkl and args.pca_model:
    print("\nLoading features + PCA for K-means clustering...")
    df_f = pd.read_pickle(args.features_pkl)
    df_f["features"] = df_f["features"].apply(
        lambda x: np.asarray(x[0], dtype=np.float32)
    )
    df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

    df_c = load_dataset(args.cov_ds)["train"].to_pandas()
    df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
    df_feat = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
    df_feat = df_feat.dropna(subset=["sic_code", "features"])
    df_feat["year"] = df_feat["year"].astype(int)

    feat_matrix = np.vstack(df_feat["features"].values)
    feat_scaler = StandardScaler().fit(feat_matrix)

    pca_loaded = joblib.load(args.pca_model)
    pca = pca_loaded["pca"] if isinstance(pca_loaded, dict) else pca_loaded

    scaled = feat_scaler.transform(feat_matrix)
    pca_features = scaled @ pca.components_.T
    df_feat["pca_features"] = list(pca_features)

    # Test years
    km_test_df = perform_kmeans_clustering_per_year(df_feat, test_years, mst_cluster_counts)
    km_test_corr_df = calculate_avg_correlation(pairs_df, km_test_df)
    kmeans_test_corr = km_test_corr_df["avg_corr"].mean()

    # All years
    km_all_df = perform_kmeans_clustering_per_year(df_feat, all_years, all_mst_cluster_counts)
    km_all_corr_df = calculate_avg_correlation(pairs_df, km_all_df)
    kmeans_all_corr = km_all_corr_df["avg_corr"].mean()

    del feat_matrix, scaled, pca_features
    import gc; gc.collect()

# ------------------------------------------------------------------
# SIC baseline
# ------------------------------------------------------------------
print("\nLoading company metadata for SIC baseline...")
df_compinfo = load_dataset(args.cov_ds)["train"].to_pandas()
df_compinfo = df_compinfo[["cik", "year", "sic_code", "__index_level_0__"]]
df_compinfo = df_compinfo.dropna(subset=["sic_code"])
df_compinfo["__index_level_0__"] = df_compinfo["__index_level_0__"].astype(str)
df_compinfo["year"] = df_compinfo["year"].astype(int)

year_SIC_cluster_df = []
for year in sorted(df_compinfo["year"].unique()):
    year_data = df_compinfo[df_compinfo["year"] == year]
    sic_clusters = (
        year_data.groupby("sic_code")["__index_level_0__"].apply(list).to_dict()
    )
    year_SIC_cluster_df.append({"year": year, "clusters": sic_clusters})
year_SIC_cluster_df = pd.DataFrame(year_SIC_cluster_df)
year_SIC_cluster_df["year"] = year_SIC_cluster_df["year"].astype(int)

sic_test_df = year_SIC_cluster_df[year_SIC_cluster_df["year"].isin(test_years)]
sic_test_corr_df = calculate_avg_correlation(pairs_df, sic_test_df)
sic_test_corr = sic_test_corr_df["avg_corr"].mean()

sic_all_corr_df = calculate_avg_correlation(pairs_df, year_SIC_cluster_df)
sic_all_corr = sic_all_corr_df["avg_corr"].mean()

pop_corr = pairs_df["correlation"].mean()

# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"RESULTS   (best θ = {best_theta:.2f}, selected on train years)")
print("=" * 70)
print(f"{'':30s} {'OOS (test)':>12s} {'All years':>12s}")
print(f"  {'Gemma SAE (MST)':30s} {test_mean_corr:>12.4f} {all_mean_corr:>12.4f}")
print(f"  {'Gemma SAE (K-means)':30s} {kmeans_test_corr:>12.4f} {kmeans_all_corr:>12.4f}")
print(f"  {'SIC code clusters':30s} {sic_test_corr:>12.4f} {sic_all_corr:>12.4f}")
print(f"  {'Population mean':30s} {pop_corr:>12.4f} {pop_corr:>12.4f}")
print("=" * 70)
