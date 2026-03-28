#!/usr/bin/env python3
"""
Ablation: supervised feature selection on PCA dimensions.
Isolates whether gains come from the SAE representation or the selection method.

Procedure:
  1. Load raw 131K SAE features
  2. Fit PCA to 4000 dims (matching parent paper)
  3. Score each PCA dim by Pearson corr of products with return correlations
  4. Select top-k, weight by score, compute dot product similarity
  5. Evaluate (Spearman rho + lift at top percentiles)

Usage (Colab):
    !python ablation_pca_supervised.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl \
        --top-k 1250 \
        --pca-dims 4000

Requires: ~20 GB RAM (Colab Pro High RAM recommended)
"""

import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from tabulate import tabulate
import gc
import time

def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--top-k", type=int, default=1250)
P.add_argument("--pca-dims", type=int, default=4000,
               help="Number of PCA components (parent paper used 4000)")
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--original-pairs-ds", default="v1ctor10/cos_sim_4000pca_exp")
args = P.parse_args()

PCTS = [0.5, 1.0, 2.0, 5.0, 10.0]

# ---- Load features ----
print("Loading features...")
df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

print("Loading metadata...")
df_c = load_dataset(args.cov_ds)["train"].to_pandas()
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)
del df_f; gc.collect()

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    print(f"  Dropping {nan_mask.sum()} rows with NaN/Inf")
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

print(f"  Feature matrix: {feat_matrix.shape[0]} companies x {feat_matrix.shape[1]} features")

# ---- Fit PCA (globally, matching parent paper) ----
print(f"\nFitting PCA to {args.pca_dims} dimensions...")
t0 = time.time()
pca = PCA(n_components=args.pca_dims, svd_solver="randomized", random_state=42)
pca_features = pca.fit_transform(feat_matrix).astype(np.float32)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

del feat_matrix; gc.collect()

# ---- Load pairs ----
print("\nLoading pairs...")
pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)

# ---- Temporal split ----
all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
train_years = set(all_years[:split_idx])
test_years = set(all_years[split_idx:])
print(f"  Train: {min(train_years)}-{max(train_years)}, Test: {min(test_years)}-{max(test_years)}")

# ---- Map companies to row indices ----
company_ids = df["__index_level_0__"].values
feat_idx_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

pairs_merged = pairs_df.merge(
    feat_idx_df.rename(columns={"__index_level_0__": "Company1", "feat_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_merged = pairs_merged.merge(
    feat_idx_df.rename(columns={"__index_level_0__": "Company2", "feat_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)

train_mask = pairs_merged["year"].isin(train_years).values
idx1_train = pairs_merged.loc[train_mask, "idx1"].values
idx2_train = pairs_merged.loc[train_mask, "idx2"].values
corr_train = pairs_merged.loc[train_mask, "correlation"].values.astype(np.float32)
print(f"  {train_mask.sum():,d} training pairs, {(~train_mask).sum():,d} test pairs")

# ---- Score each PCA dimension ----
# For each dim j: Pearson corr between (pca[i1,j] * pca[i2,j]) and return_corr
print(f"\nScoring {args.pca_dims} PCA dimensions (Pearson of products)...")
t0 = time.time()
scores = np.zeros(args.pca_dims, dtype=np.float64)

corr_train_demean = corr_train - corr_train.mean()
corr_train_std = corr_train.std()

for j in range(args.pca_dims):
    products = pca_features[idx1_train, j] * pca_features[idx2_train, j]
    prod_demean = products - products.mean()
    prod_std = prod_demean.std()
    if prod_std > 0:
        scores[j] = (prod_demean * corr_train_demean).mean() / (prod_std * corr_train_std)
    if (j + 1) % 500 == 0:
        print(f"  {j+1}/{args.pca_dims} scored ({time.time()-t0:.1f}s)")

n_positive = (scores > 0).sum()
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Score range: {scores.min():.6f} to {scores.max():.6f}")
print(f"  Positive scores: {n_positive} / {args.pca_dims}")

# ---- Select top-k ----
ranked = np.argsort(scores)[::-1]
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)
print(f"\n  Selected {n_selected} PCA dimensions (requested {args.top_k})")

if n_selected == 0:
    print("ERROR: No PCA dimensions scored positively. Exiting.")
    exit(1)

# ---- Compute similarities (score-weighted dot product, alpha=0) ----
print("\nComputing similarities...")
selected_pca = pca_features[:, selected].copy()
weights = scores[selected].astype(np.float32)
selected_pca *= weights[np.newaxis, :]

idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
correlations = pairs_merged["correlation"].values.astype(np.float32)

sims = np.empty(len(pairs_merged), dtype=np.float32)
batch = 500_000
for s in range(0, len(pairs_merged), batch):
    e = min(s + batch, len(pairs_merged))
    i1, i2 = idx1[s:e], idx2[s:e]
    sims[s:e] = (selected_pca[i1] * selected_pca[i2]).sum(1)

# ---- Evaluate ----
rho_all, pval_all = spearmanr(sims, correlations)

test_mask_eval = pairs_merged["year"].isin(test_years).values
test_sims = sims[test_mask_eval]
test_corrs = correlations[test_mask_eval]
test_sorted = np.argsort(test_sims)[::-1]

rho_test, pval_test = spearmanr(test_sims, test_corrs)

# ---- Report ----
print(f"\n{'='*70}")
print(f"ABLATION: SUPERVISED PCA (dims={args.pca_dims}, selected={n_selected})")
print(f"{'='*70}")
print(f"\nSpearman rho (all years): {rho_all:.4f} (p={pval_all:.2e})")
print(f"Spearman rho (OOS only):  {rho_test:.4f} (p={pval_test:.2e})")

print(f"\nLift at top-k (OOS {min(test_years)}-{max(test_years)}, "
      f"{len(test_sims):,d} pairs):")
rows = []
for pct in PCTS:
    n_top = max(1, int(len(test_sims) * pct / 100.0))
    top_mean = test_corrs[test_sorted[:n_top]].mean()
    rows.append([f"top {pct:.1f}%", f"{top_mean:.4f}"])

print(tabulate(rows, headers=["Cutoff", "Mean return corr"], tablefmt="simple_outline"))
print(f"\nPopulation mean: {correlations.mean():.4f}")

# ---- Also report unsupervised PCA cosine (matching parent paper exactly) ----
print(f"\n--- Unsupervised PCA cosine (all {args.pca_dims} dims, no selection) ---")
norms_pca = np.linalg.norm(pca_features, axis=1).clip(min=1e-10)

cos_unsup = np.empty(len(pairs_merged), dtype=np.float32)
for s in range(0, len(pairs_merged), batch):
    e = min(s + batch, len(pairs_merged))
    i1, i2 = idx1[s:e], idx2[s:e]
    dot = (pca_features[i1] * pca_features[i2]).sum(1)
    cos_unsup[s:e] = dot / (norms_pca[i1] * norms_pca[i2])

rho_unsup, pval_unsup = spearmanr(cos_unsup, correlations)
test_cos_unsup = cos_unsup[test_mask_eval]
test_sorted_unsup = np.argsort(test_cos_unsup)[::-1]

print(f"Spearman rho (all years): {rho_unsup:.4f}")
print("Lift at top-k (OOS):")
for pct in PCTS:
    n_top = max(1, int(len(test_cos_unsup) * pct / 100.0))
    top_mean = test_corrs[test_sorted_unsup[:n_top]].mean()
    print(f"  top {pct:.1f}%: {top_mean:.4f}")

print(f"\n{'='*70}")
print("COMPARE AGAINST YOUR SAE RESULTS (from evaluate.py):")
print("  SAE supervised (k=1250, α=0): Spearman=0.1826, OOS top-1%=0.3762")
print("  Parent paper PCA cosine:       Spearman=0.0217, OOS top-1%=0.1598")
print(f"{'='*70}")
