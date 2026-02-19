#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--pca-components", type=int, default=4000)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--original-pairs-ds", default="v1ctor10/cos_sim_4000pca_exp")
P.add_argument("--out-pairs", default="data/gemma_pairs.pkl")
P.add_argument("--out-pca", default="data/gemma_pca_model.pkl")
args = P.parse_args()

os.makedirs(os.path.dirname(args.out_pairs) or ".", exist_ok=True)

print("Loading Gemma features...")
df_f = pd.read_pickle(args.features_pkl)

sample_feat = df_f["features"].iloc[0]
if isinstance(sample_feat, list):
    if len(sample_feat) == 1 and hasattr(sample_feat[0], '__len__'):
        df_f["features"] = df_f["features"].apply(lambda x: np.asarray(x[0], dtype=np.float32))
    else:
        df_f["features"] = df_f["features"].apply(lambda x: np.asarray(x, dtype=np.float32))
elif isinstance(sample_feat, np.ndarray):
    if sample_feat.ndim == 2 and sample_feat.shape[0] == 1:
        df_f["features"] = df_f["features"].apply(lambda x: x[0].astype(np.float32))
    else:
        df_f["features"] = df_f["features"].apply(lambda x: np.asarray(x, dtype=np.float32))
else:
    df_f["features"] = df_f["features"].apply(lambda x: np.asarray(x, dtype=np.float32))

print("Loading company metadata...")
df_c = load_dataset(args.cov_ds)["train"].to_pandas()

df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)
print(f"  Matched {len(df)} companies with features + metadata")

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    print(f"  Dropping {nan_mask.sum()} rows with NaN/Inf features")
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

if len(feat_matrix) == 0:
    raise RuntimeError("No valid feature rows remain after NaN removal.")

feat_dim = feat_matrix.shape[1]

print("Fitting StandardScaler + PCA...")
scaler = StandardScaler().fit(feat_matrix)
scaled = scaler.transform(feat_matrix)

n_components = min(args.pca_components, feat_dim, feat_matrix.shape[0])
pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(scaled)
print(
    f"  PCA: {feat_dim} -> {n_components} components, "
    f"explained variance: {pca.explained_variance_ratio_.sum():.4f}"
)

pca_bundle = {"pca": pca, "scaler": scaler}
joblib.dump(pca_bundle, args.out_pca)
print(f"Saved PCA bundle to {args.out_pca}")

pca_df = pd.DataFrame({
    "__index_level_0__": df["__index_level_0__"].values,
    "year": df["year"].values,
    "pca_idx": np.arange(len(df)),
})

print("Loading original pairs dataset (for company pairs + stock correlations)...")
pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)
print(f"  Loaded {len(pairs_df)} original pairs")

print("Matching pairs to PCA features...")
pairs_df = pairs_df.merge(
    pca_df.rename(columns={"__index_level_0__": "Company1", "pca_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_df = pairs_df.merge(
    pca_df.rename(columns={"__index_level_0__": "Company2", "pca_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)
print(f"  {len(pairs_df)} pairs have both companies with Gemma features")

print("Computing cosine similarities in PCA space (batched)...")
norms = np.linalg.norm(pca_features, axis=1).clip(min=1e-10)

idx1 = pairs_df["idx1"].values
idx2 = pairs_df["idx2"].values
cos_sims = np.empty(len(pairs_df), dtype=np.float32)

batch = 500_000
for s in range(0, len(pairs_df), batch):
    e = min(s + batch, len(pairs_df))
    i1, i2 = idx1[s:e], idx2[s:e]
    cos_sims[s:e] = (
        (pca_features[i1] * pca_features[i2]).sum(1)
        / (norms[i1] * norms[i2])
    )

pairs_df["cosine_similarity"] = cos_sims
pairs_df = pairs_df.drop(columns=["idx1", "idx2"])

pairs_df.to_pickle(args.out_pairs)
print(f"Saved {len(pairs_df)} pairs with Gemma cosine similarities to {args.out_pairs}")
print(f"  Mean cosine similarity: {cos_sims.mean():.4f}")
print(f"  Mean stock correlation:  {pairs_df['correlation'].mean():.4f}")
