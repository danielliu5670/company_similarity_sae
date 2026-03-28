#!/usr/bin/env python3
"""
Ablation: unsupervised SAE baseline (all features, no selection). GPU-accelerated.
Tests whether the raw SAE representation outperforms PCA even without supervision.

Strategy: feat_matrix (27K x 131K) does not fit on GPU.
  - Outer loop: chunk over features (4096 at a time, ~454 MB per chunk)
  - Inner loop: batch over pairs (100K at a time)
  - Accumulate partial dot products on GPU.

Usage (Colab with GPU runtime + High RAM):
    !pip install cupy-cuda12x tabulate
    !python ablation_unsupervised_sae.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl
"""

import argparse
import numpy as np
import pandas as pd
import cupy as cp
from datasets import load_dataset
from scipy.stats import spearmanr
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


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--feat-chunk", type=int, default=4096,
               help="Features per GPU chunk (reduce to 2048 if OOM)")
P.add_argument("--pair-batch", type=int, default=100_000,
               help="Pairs per GPU batch (reduce to 50000 if OOM)")
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
print(f"  Feature matrix: {n_companies} x {feat_dim}")
print(f"  RAM usage: ~{feat_matrix.nbytes / 1e9:.1f} GB")

# Precompute norms on CPU (just a 1D array)
print("Computing L2 norms...")
norms = np.linalg.norm(feat_matrix, axis=1).astype(np.float32)
norms = np.clip(norms, 1e-10, None)
print(f"  Norm range: {norms.min():.2f} to {norms.max():.2f}, "
      f"median: {np.median(norms):.2f}")

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

idx1 = pairs_merged["idx1"].values.astype(np.int64)
idx2 = pairs_merged["idx2"].values.astype(np.int64)
correlations = pairs_merged["correlation"].values.astype(np.float32)
n_pairs = len(pairs_merged)
print(f"  {n_pairs:,d} matched pairs")

# ==================================================================
# Compute dot products: double-chunked over features and pairs.
#
# dot(a,b) = sum_j feat[a,j] * feat[b,j]
#
# Outer loop: load feat_matrix[:, j_start:j_end] to GPU once.
# Inner loop: for each pair batch, index into the chunk and accumulate.
# Dot products accumulate on GPU to avoid CPU-GPU transfers per batch.
# ==================================================================
print(f"\nComputing dot products on GPU...")
print(f"  feat_chunk={args.feat_chunk}, pair_batch={args.pair_batch}")
t0 = time.time()

idx1_gpu = cp.asarray(idx1)
idx2_gpu = cp.asarray(idx2)
dot_sims_gpu = cp.zeros(n_pairs, dtype=cp.float32)

feat_chunk = args.feat_chunk
pair_batch = args.pair_batch
n_feat_chunks = (feat_dim + feat_chunk - 1) // feat_chunk
n_pair_batches = (n_pairs + pair_batch - 1) // pair_batch

for fi in tqdm(range(n_feat_chunks), desc="Feature chunks"):
    fs = fi * feat_chunk
    fe = min(fs + feat_chunk, feat_dim)

    feat_gpu = cp.asarray(feat_matrix[:, fs:fe])

    for s in range(0, n_pairs, pair_batch):
        e = min(s + pair_batch, n_pairs)
        a = feat_gpu[idx1_gpu[s:e]]
        b = feat_gpu[idx2_gpu[s:e]]
        dot_sims_gpu[s:e] += (a * b).sum(axis=1)
        del a, b

    del feat_gpu
    cp.get_default_memory_pool().free_all_blocks()

dot_sims = cp.asnumpy(dot_sims_gpu)
del dot_sims_gpu
cp.get_default_memory_pool().free_all_blocks()

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# Cosine = dot / (norm1 * norm2)
cos_sims = dot_sims / (norms[idx1] * norms[idx2])

del idx1_gpu, idx2_gpu
cp.get_default_memory_pool().free_all_blocks()

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
print("  SAE supervised (k=1250, a=0):  Spearman=0.1826, top-1%=0.3762")
print("  Parent paper PCA cosine:       Spearman=0.0217, top-1%=0.1598")
print("  SIC baseline:                                    top-1%=0.2835")
print(f"{'='*60}")
