#!/usr/bin/env python3
"""
Supervised feature selection + cosine similarity.
Replaces unsupervised PCA with feature scoring against return correlations.

Usage (Colab):
    !python compute_similarities_gemma.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/gemma_features.pkl \
        --top-k 500 \
        --out-pairs /content/drive/MyDrive/company_similarity_sae/data/gemma_pairs.pkl \
        --out-model /content/drive/MyDrive/company_similarity_sae/data/gemma_selection_model.pkl
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--top-k", type=int, default=500,
               help="Number of top-scoring features to retain")
P.add_argument("--min-support", type=int, default=50,
               help="Minimum co-active pairs to score a feature")
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--original-pairs-ds", default="v1ctor10/cos_sim_4000pca_exp")
P.add_argument("--out-pairs", default="data/gemma_pairs.pkl")
P.add_argument("--out-model", default="data/gemma_selection_model.pkl")
args = P.parse_args()

os.makedirs(os.path.dirname(args.out_pairs) or ".", exist_ok=True)

# ------------------------------------------------------------------
# Load and unwrap features (identical to original)
# ------------------------------------------------------------------
print("Loading Gemma features...")
df_f = pd.read_pickle(args.features_pkl)

sample_feat = df_f["features"].iloc[0]
print(f"  Feature type: {type(sample_feat)}, ", end="")
if isinstance(sample_feat, (list, np.ndarray)):
    flat = sample_feat
    while isinstance(flat, list) and len(flat) == 1 and isinstance(flat[0], (list, np.ndarray)):
        flat = flat[0]
    print(f"unwrapped type: {type(flat)}, len: {len(flat) if hasattr(flat, '__len__') else 'N/A'}")


def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


df_f["features"] = df_f["features"].apply(unwrap_feature)
print(f"  Final feature shape: {df_f['features'].iloc[0].shape}")

# ------------------------------------------------------------------
# Merge with company metadata (identical to original)
# ------------------------------------------------------------------
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
print(f"  Feature matrix: {feat_matrix.shape[0]} companies x {feat_dim} features")

# ------------------------------------------------------------------
# Load pairs dataset
# ------------------------------------------------------------------
print("Loading original pairs dataset...")
pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)
print(f"  Loaded {len(pairs_df)} original pairs")

# ------------------------------------------------------------------
# Temporal split (must match threshold_gemma.py)
# ------------------------------------------------------------------
all_years = sorted(pairs_df["year"].unique())
n_total = len(all_years)
split_idx = int(0.75 * n_total)
train_years = set(all_years[:split_idx])
test_years = set(all_years[split_idx:])

print(f"  Train years: {min(train_years)}-{max(train_years)} ({len(train_years)} years)")
print(f"  Test years:  {min(test_years)}-{max(test_years)} ({len(test_years)} years)")

# ------------------------------------------------------------------
# Map companies to row indices in feat_matrix
# ------------------------------------------------------------------
company_ids = df["__index_level_0__"].values
company_to_idx = {c: i for i, c in enumerate(company_ids)}

# ------------------------------------------------------------------
# Build index arrays for TRAINING pairs only
# ------------------------------------------------------------------
print("Preparing training pairs for feature scoring...")
train_pairs = pairs_df[pairs_df["year"].isin(train_years)].copy()

# Keep only pairs where both companies have features
valid_mask = (
    train_pairs["Company1"].isin(company_to_idx)
    & train_pairs["Company2"].isin(company_to_idx)
)
train_pairs = train_pairs[valid_mask].reset_index(drop=True)

idx1_train = train_pairs["Company1"].map(company_to_idx).values.astype(np.int64)
idx2_train = train_pairs["Company2"].map(company_to_idx).values.astype(np.int64)
corr_train = train_pairs["correlation"].values.astype(np.float32)
pop_mean = corr_train.mean()

print(f"  {len(train_pairs)} training pairs (population mean corr: {pop_mean:.4f})")

# ------------------------------------------------------------------
# Binarize features: active = nonzero after JumpReLU
# ------------------------------------------------------------------
print("Binarizing feature matrix...")
binary_matrix = (feat_matrix > 0)  # bool, ~1.6 GB for 24K x 65K

# ------------------------------------------------------------------
# Score each feature against return correlations
# ------------------------------------------------------------------
print(f"Scoring {feat_dim} features (min support = {args.min_support} pairs)...")
scores = np.zeros(feat_dim, dtype=np.float64)
support = np.zeros(feat_dim, dtype=np.int64)

chunk_size = 128
n_chunks = (feat_dim + chunk_size - 1) // chunk_size

for ci in tqdm(range(n_chunks), desc="Scoring features (chunked)"):
    j_start = ci * chunk_size
    j_end = min(j_start + chunk_size, feat_dim)

    cols = binary_matrix[:, j_start:j_end]        # (n_companies, chunk) bool
    b1 = cols[idx1_train]                          # (n_pairs, chunk) bool
    b2 = cols[idx2_train]                          # (n_pairs, chunk) bool
    co = b1 & b2
    del b1, b2

    counts = co.sum(axis=0)                        # (chunk,)
    corr_sums = corr_train @ co.astype(np.float32) # (chunk,)
    del co

    for local_j in range(j_end - j_start):
        j = j_start + local_j
        n = counts[local_j]
        if n >= args.min_support:
            scores[j] = corr_sums[local_j] / n - pop_mean
            support[j] = int(n)

# ------------------------------------------------------------------
# Select top-k features with positive scores
# ------------------------------------------------------------------
ranked = np.argsort(scores)[::-1]
# Keep at most top_k, but only those with score > 0
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)  # sort by index for stable column ordering

print(f"\nSelected {len(selected)} features (requested top {args.top_k})")
if len(selected) == 0:
    raise RuntimeError("No features scored positively. Check data or lower --min-support.")

print(f"  Score range of selected: {scores[selected].min():.6f} to {scores[selected].max():.6f}")
print(f"  Support range: {support[selected].min()} to {support[selected].max()} pairs")
print(f"  Top 10 feature indices: {selected[np.argsort(scores[selected])[::-1]][:10]}")
print(f"  Top 10 scores:          {np.sort(scores[selected])[::-1][:10]}")

# Free the binary matrix
del binary_matrix

# ------------------------------------------------------------------
# Scale selected features
# ------------------------------------------------------------------
print("Extracting selected features...")
selected_features = feat_matrix[:, selected]
scaled_features = selected_features
scaler = None

# ------------------------------------------------------------------
# Save model (selected indices + scaler + scores for interpretability)
# ------------------------------------------------------------------
model_bundle = {
    "selected_indices": selected,
    "scaler": scaler,
    "scores": scores,
    "support": support,
    "pop_mean_train": pop_mean,
    "train_years": sorted(train_years),
    "test_years": sorted(test_years),
}
joblib.dump(model_bundle, args.out_model)
print(f"Saved selection model to {args.out_model}")

# ------------------------------------------------------------------
# Match ALL pairs to feature indices (same logic as original)
# ------------------------------------------------------------------
feat_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

print("Matching pairs to selected features...")
pairs_df = pairs_df.merge(
    feat_df.rename(columns={"__index_level_0__": "Company1", "feat_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_df = pairs_df.merge(
    feat_df.rename(columns={"__index_level_0__": "Company2", "feat_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)
print(f"  {len(pairs_df)} pairs have both companies with features")

# ------------------------------------------------------------------
# Compute cosine similarities in selected-feature space (batched)
# ------------------------------------------------------------------
print(f"Computing cosine similarities in {len(selected)}-D selected space...")
norms = np.linalg.norm(scaled_features, axis=1).clip(min=1e-10)

idx1 = pairs_df["idx1"].values
idx2 = pairs_df["idx2"].values
cos_sims = np.empty(len(pairs_df), dtype=np.float32)

batch = 500_000
for s in range(0, len(pairs_df), batch):
    e = min(s + batch, len(pairs_df))
    i1, i2 = idx1[s:e], idx2[s:e]
    cos_sims[s:e] = (
        (scaled_features[i1] * scaled_features[i2]).sum(1)
        / (norms[i1] * norms[i2])
    )

pairs_df["cosine_similarity"] = cos_sims
pairs_df = pairs_df.drop(columns=["idx1", "idx2"])

# ------------------------------------------------------------------
# Quick diagnostic: Spearman correlation
# ------------------------------------------------------------------
from scipy.stats import spearmanr
rho, pval = spearmanr(cos_sims, pairs_df["correlation"].values)
print(f"\n  Spearman rho (similarity vs return correlation): {rho:.4f}  (p={pval:.2e})")

# ------------------------------------------------------------------
# Save pairs
# ------------------------------------------------------------------
pairs_df.to_pickle(args.out_pairs)
print(f"\nSaved {len(pairs_df)} pairs to {args.out_pairs}")
print(f"  Mean cosine similarity: {cos_sims.mean():.4f}")
print(f"  Mean stock correlation:  {pairs_df['correlation'].mean():.4f}")
