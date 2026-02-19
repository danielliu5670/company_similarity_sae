#!/usr/bin/env python3

import argparse
import os
import pickle
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
    for year in tqdm(years, desc="MST clustering"):
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

        results.append({"year": year, "clusters": cluster_dict})

    return pd.DataFrame(results)


def perform_kmeans_clustering_per_year(feat_df, years, target_k_per_year):
    results = []
    for year in tqdm(years, desc="K-means clustering"):
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


def calculate_avg_correlation(pairs_df, cluster_df, label):
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
            avg_correlations.append(
                {"year": year, f"{label}AvgCorrelation": np.mean(cluster_stats)}
            )
        else:
            avg_correlations.append({"year": year, f"{label}AvgCorrelation": np.nan})

    return pd.DataFrame(avg_correlations)


P = argparse.ArgumentParser()
P.add_argument("--pairs-pkl", required=True)
P.add_argument("--threshold", type=float, default=-2.7)
P.add_argument("--out-clusters", default="data/gemma_year_cluster_dfC-CD.pkl")
P.add_argument("--out-clusters-kmeans", default="data/gemma_year_cluster_dfC-CD_kmeans.pkl")
P.add_argument("--features-pkl", default=None)
P.add_argument("--pca-model", default=None)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
args = P.parse_args()

os.makedirs(os.path.dirname(args.out_clusters) or ".", exist_ok=True)

print("Loading pairs...")
pairs_df = pd.read_pickle(args.pairs_pkl)
pairs_df["year"] = pairs_df["year"].astype(int)

pairs_df["cosine_distance"] = 1 - pairs_df["cosine_similarity"]

all_years = sorted(pairs_df["year"].unique())
n_total_years = len(all_years)
split_end = int(0.75 * n_total_years)

scaler = StandardScaler()
train_mask = pairs_df["year"].isin(all_years[:split_end])
scaler.fit(pairs_df.loc[train_mask, ["cosine_distance"]])
pairs_df["cosine_distance_scaled"] = scaler.transform(pairs_df[["cosine_distance"]])

print(f"MST clustering {len(all_years)} years with threshold={args.threshold}...")
year_cluster_df = perform_clustering_per_year(
    pairs_df, all_years, threshold=args.threshold
)
year_cluster_df["year"] = year_cluster_df["year"].astype(int)
year_cluster_df["clusters"] = year_cluster_df["clusters"].apply(
    lambda x: {k: v for k, v in x.items() if k != -1}
)

mst_cluster_counts = {}
for _, row in year_cluster_df.iterrows():
    k = len([c for c in row["clusters"].values() if len(c) > 1])
    mst_cluster_counts[row["year"]] = max(k, 2)

total_clusters = sum(mst_cluster_counts.values())
print(f"  Total MST clusters (size > 1): {total_clusters}")

year_cluster_df.to_pickle(args.out_clusters)
print(f"Saved MST clusters to {args.out_clusters}")

year_kmeans_cluster_df = None
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

    print(f"  K-means clustering with k matched to MST cluster count per year...")
    year_kmeans_cluster_df = perform_kmeans_clustering_per_year(
        df_feat, all_years, mst_cluster_counts
    )
    year_kmeans_cluster_df["year"] = year_kmeans_cluster_df["year"].astype(int)

    year_kmeans_cluster_df.to_pickle(args.out_clusters_kmeans)
    print(f"Saved K-means clusters to {args.out_clusters_kmeans}")

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

gemma_mst_corr = calculate_avg_correlation(pairs_df, year_cluster_df, "GemmaMST")
sic_corr = calculate_avg_correlation(pairs_df, year_SIC_cluster_df, "SIC")

results = pd.merge(gemma_mst_corr, sic_corr, on="year", how="outer")

if year_kmeans_cluster_df is not None:
    gemma_km_corr = calculate_avg_correlation(
        pairs_df, year_kmeans_cluster_df, "GemmaKMeans"
    )
    results = pd.merge(results, gemma_km_corr, on="year", how="outer")

pop_corr = pairs_df["correlation"].mean()

print("\n" + "=" * 65)
print("CLUSTERING QUALITY (Mean Correlation Within Clusters)")
print("=" * 65)
print(f"  Gemma SAE (MST):      {results['GemmaMSTAvgCorrelation'].mean():.4f}")
if year_kmeans_cluster_df is not None:
    print(f"  Gemma SAE (K-means):  {results['GemmaKMeansAvgCorrelation'].mean():.4f}")
print(f"  SIC code clusters:    {results['SICAvgCorrelation'].mean():.4f}")
print(f"  Population mean:      {pop_corr:.4f}")
print("=" * 65)
