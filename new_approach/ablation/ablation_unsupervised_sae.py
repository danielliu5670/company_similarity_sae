#!/usr/bin/env python3
"""
Ablation: unsupervised SAE baseline (all features, no selection).
Tests whether the raw SAE representation outperforms PCA even without supervision.

Computes pairwise dot product and cosine similarity over all ~131K SAE features.
No scoring, no selection, no weighting.

Usage (Colab):
    !python ablation_unsupervised_sae.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl

IMPORTANT: Requires ~16-18 GB RAM. Use Colab Pro with High RAM runtime.
The computation over 131K features x 15M pairs takes ~15-30 minutes.
"""

import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
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
P.add_argument("--pair-batch", type=int, default=50_000,
               help="Pairs per batch (reduce if OOM)")
P.add_argument("--feat-chunk", type=int, default=4000,
               help="Features per chunk (reduce if OOM)")
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
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

n_companies, feat_dim = feat_matrix.shape
print(f"  Feature matrix: {n_companies} companies x {feat_dim} features")
print(f"  Memory: ~{feat_matrix.nbytes / 1e9:.1f} GB")

# ---- Precompute norms ----
print("Computing L2 norms...")
norms = np.linalg.norm(feat_matrix, axis=1).astype(np.float32)
norms = np.clip(norms, 1e-10, None)
print(f"  Norm range: {norms.min():.2f} to {norms.max():.2f}")
print(f"  Median norm: {np.median(norms):.2f}")

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
test_years = set(all_years[split_idx:])

# ---- Map companies to indices ----
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

idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
correlations = pairs_merged["correlation"].values.astype(np.float32)
n_pairs = len(pairs_merged)
print(f"  {n_pairs:,d} matched pairs")

# ---- Compute dot products (chunked over both pairs and features) ----
print(f"\nComputing dot products over all {feat_dim} features...")
print(f"  pair_batch={args.pair_batch}, feat_chunk={args.feat_chunk}")
t0 = time.time()

dot_sims = np.zeros(n_pairs, dtype=np.float64)
pair_batch = args.pair_batch
feat_chunk = args.feat_chunk
n_pair_batches = (n_pairs + pair_batch - 1) // pair_batch

for bi, s in enumerate(range(0, n_pairs, pair_batch)):
    e = min(s + pair_batch, n_pairs)
    i1, i2 = idx1[s:e], idx2[s:e]
    batch_dot = np.zeros(e - s, dtype=np.float64)

    for fs in range(0, feat_dim, feat_chunk):
        fe = min(fs + feat_chunk, feat_dim)
        a = feat_matrix[i1, fs:fe]
        b = feat_matrix[i2, fs:fe]
        batch_dot += (a.astype(np.float64) * b.astype(np.float64)).sum(axis=1)

    dot_sims[s:e] = batch_dot

    if (bi + 1) % 20 == 0 or bi == 0:
        elapsed = time.time() - t0
        rate = (bi + 1) / elapsed
        remaining = (n_pair_batches - bi - 1) / rate
        print(f"  Batch {bi+1}/{n_pair_batches} "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

dot_sims = dot_sims.astype(np.float32)
elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# ---- Cosine similarity = dot / (norm1 * norm2) ----
cos_sims = dot_sims / (norms[idx1] * norms[idx2])

# ---- Evaluation helper ----
def evaluate(sims, label):
    rho, pval = spearmanr(sims, correlations)
    test_mask = pairs_merged["year"].isin(test_years).values
    test_sims = sims[test_mask]
    test_corrs = correlations[test_mask]
    test_sorted = np.argsort(test_sims)[::-1]

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Spearman rho (all years): {rho:.4f} (p={pval:.2e})")

    rows = []
    for pct in PCTS:
        n_top = max(1, int(len(test_sims) * pct / 100.0))
        top_mean = test_corrs[test_sorted[:n_top]].mean()
        rows.append([f"top {pct:.1f}%", f"{top_mean:.4f}"])

    print(f"\nLift at top-k (OOS {min(test_years)}-{max(test_years)}, "
          f"{len(test_sims):,d} pairs):")
    print(tabulate(rows, headers=["Cutoff", "Mean return corr"],
                   tablefmt="simple_outline"))

evaluate(dot_sims, "UNSUPERVISED SAE: all features, dot product")
evaluate(cos_sims, "UNSUPERVISED SAE: all features, cosine similarity")

print(f"\nPopulation mean: {correlations.mean():.4f}")

print(f"\n{'='*60}")
print("COMPARE:")
print("  Your SAE supervised (k=1250, α=0):  Spearman=0.1826, top-1%=0.3762")
print("  Parent paper PCA cosine:             Spearman=0.0217, top-1%=0.1598")
print("  SIC baseline:                                         top-1%=0.2835")
print(f"{'='*60}")
