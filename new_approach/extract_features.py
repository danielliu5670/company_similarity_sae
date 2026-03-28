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
import cupy as cp

P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--top-k", type=int, nargs="+", default=[500],
               help="Number(s) of top-scoring features to retain (space-separated for sweep)")
P.add_argument("--min-support", type=int, default=50,
               help="Minimum co-active pairs to score a feature")
P.add_argument("--score-weight", action="store_true",
               help="Multiply each selected feature by its score before cosine sim")
P.add_argument("--load-model", default=None,
               help="Load pre-computed scores from this model file; skip scoring loop")
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
# Feature scoring: either load pre-computed or run from scratch
# ------------------------------------------------------------------
if args.load_model is not None:
    print(f"\nLoading pre-computed scores from {args.load_model}...")
    saved = joblib.load(args.load_model)
    scores = saved["scores"]
    support = saved["support"]
    print(f"  Loaded scores for {len(scores)} features")
else:
    # ------------------------------------------------------------------
    # Binarize features: active = nonzero after JumpReLU
    # ------------------------------------------------------------------
    print("Binarizing feature matrix...")
    binary_matrix = (feat_matrix > 0)

    # ------------------------------------------------------------------
    # Score each feature against return correlations (GPU)
    # ------------------------------------------------------------------
    print(f"Scoring {feat_dim} features (min support = {args.min_support} pairs)...")
    scores = np.zeros(feat_dim, dtype=np.float64)
    support = np.zeros(feat_dim, dtype=np.int64)

    binary_gpu = cp.asarray(binary_matrix)
    idx1_gpu = cp.asarray(idx1_train)
    idx2_gpu = cp.asarray(idx2_train)
    corr_gpu = cp.asarray(corr_train)

    chunk_size = 512
    n_chunks = (feat_dim + chunk_size - 1) // chunk_size

    for ci in tqdm(range(n_chunks), desc="Scoring features (GPU)"):
        j_start = ci * chunk_size
        j_end = min(j_start + chunk_size, feat_dim)

        cols = binary_gpu[:, j_start:j_end]
        b1 = cols[idx1_gpu]
        b2 = cols[idx2_gpu]
        co = b1 & b2
        del b1, b2

        counts = co.sum(axis=0)
        corr_sums = corr_gpu @ co.astype(cp.float32)
        del co

        counts_cpu = cp.asnumpy(counts)
        corr_sums_cpu = cp.asnumpy(corr_sums)
        j_indices = np.arange(j_start, j_end)
        valid = counts_cpu >= args.min_support
        if valid.any():
            scores[j_indices[valid]] = corr_sums_cpu[valid] / counts_cpu[valid] - pop_mean
            support[j_indices[valid]] = counts_cpu[valid]

    del binary_gpu, idx1_gpu, idx2_gpu, corr_gpu
    cp.get_default_memory_pool().free_all_blocks()

# ------------------------------------------------------------------
# Match all pairs to feature indices 
# ------------------------------------------------------------------
feat_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

print("Matching pairs to feature indices...")
pairs_merged = pairs_df.merge(
    feat_df.rename(columns={"__index_level_0__": "Company1", "feat_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_merged = pairs_merged.merge(
    feat_df.rename(columns={"__index_level_0__": "Company2", "feat_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)
print(f"  {len(pairs_merged)} pairs have both companies with features")

from scipy.stats import spearmanr

idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
correlations = pairs_merged["correlation"].values

# ------------------------------------------------------------------
# Sweep over top-k values
# ------------------------------------------------------------------
ranked = np.argsort(scores)[::-1]
summary_rows = []

for top_k in args.top_k:
    print(f"\n{'='*70}")
    print(f"  top-k = {top_k}")
    print(f"{'='*70}")

    selected = ranked[:top_k]
    selected = selected[scores[selected] > 0]
    selected = np.sort(selected)

    print(f"  Selected {len(selected)} features (requested {top_k})")
    if len(selected) == 0:
        print("  WARNING: No features scored positively, skipping.")
        continue

    print(f"  Score range: {scores[selected].min():.6f} to {scores[selected].max():.6f}")
    print(f"  Top 10 feature indices: {selected[np.argsort(scores[selected])[::-1]][:10]}")
    print(f"  Top 10 scores:          {np.sort(scores[selected])[::-1][:10]}")

    # Extract and optionally score-weight
    selected_features = feat_matrix[:, selected].copy()
    if args.score_weight:
        weights = scores[selected].astype(np.float32)
        selected_features *= weights[np.newaxis, :]

    norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10)

    # Compute cosine similarities
    cos_sims = np.empty(len(pairs_merged), dtype=np.float32)
    batch = 500_000
    for s in range(0, len(pairs_merged), batch):
        e = min(s + batch, len(pairs_merged))
        i1, i2 = idx1[s:e], idx2[s:e]
        cos_sims[s:e] = (
            (selected_features[i1] * selected_features[i2]).sum(1)
            / (norms[i1] * norms[i2])
        )

    rho, pval = spearmanr(cos_sims, correlations)
    print(f"\n  Spearman rho: {rho:.4f}  (p={pval:.2e})")

    # Precision-at-k (all pairs)
    print(f"\n  Precision-at-k (all pairs, {len(cos_sims):,d} total):")
    sorted_indices = np.argsort(cos_sims)[::-1]
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0]:
        n_top = max(1, int(len(cos_sims) * pct / 100.0))
        top_idx = sorted_indices[:n_top]
        top_mean_corr = correlations[top_idx].mean()
        print(f"    Top {pct:5.1f}% ({n_top:>8,d} pairs): mean return corr = {top_mean_corr:.4f}")

    # Precision-at-k (OOS only)
    test_mask = pairs_merged["year"].isin(test_years).values
    test_sims = cos_sims[test_mask]
    test_corrs = correlations[test_mask]
    test_sorted = np.argsort(test_sims)[::-1]

    oos_top1_corr = np.nan
    print(f"\n  Precision-at-k (OOS test years, {len(test_sims):,d} pairs):")
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0]:
        n_top = max(1, int(len(test_sims) * pct / 100.0))
        top_idx = test_sorted[:n_top]
        top_mean_corr = test_corrs[top_idx].mean()
        print(f"    Top {pct:5.1f}% ({n_top:>8,d} pairs): mean return corr = {top_mean_corr:.4f}")
        if pct == 1.0:
            oos_top1_corr = top_mean_corr

    summary_rows.append({
        "top_k": top_k, "n_selected": len(selected),
        "spearman_rho": rho, "oos_top1pct_corr": oos_top1_corr,
    })

    # Save pairs for this k
    out_pairs_df = pairs_merged.drop(columns=["idx1", "idx2"]).copy()
    out_pairs_df["cosine_similarity"] = cos_sims

    if len(args.top_k) == 1:
        out_path = args.out_pairs
    else:
        base, ext = os.path.splitext(args.out_pairs)
        out_path = f"{base}_k{top_k}{ext}"

    out_pairs_df.to_pickle(out_path)
    print(f"  Saved {len(out_pairs_df)} pairs to {out_path}")

# ------------------------------------------------------------------
# Save model (scores array for future --load-model runs)
# ------------------------------------------------------------------
model_bundle = {
    "selected_indices": selected,
    "scaler": None,
    "scores": scores,
    "support": support,
    "pop_mean_train": pop_mean,
    "train_years": sorted(train_years),
    "test_years": sorted(test_years),
    "score_weighted": args.score_weight,
    "top_k_used": len(selected),
}
joblib.dump(model_bundle, args.out_model)
print(f"\nSaved selection model to {args.out_model}")

# ------------------------------------------------------------------
# Summary table (only printed when sweeping multiple k values)
# ------------------------------------------------------------------
if len(summary_rows) > 1:
    print(f"\n{'='*70}")
    print("K-SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'top_k':>8s}  {'n_sel':>6s}  {'rho':>8s}  {'OOS top1%':>10s}")
    for r in summary_rows:
        print(f"  {r['top_k']:>8d}  {r['n_selected']:>6d}  {r['spearman_rho']:>8.4f}  {r['oos_top1pct_corr']:>10.4f}")
    print(f"{'='*70}")