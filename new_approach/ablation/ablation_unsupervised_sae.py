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
from tabulate import tabulate, SEPARATING_LINE
import gc

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
P.add_argument("--sae-spearman-oos", type=float, default=None,
               help="OOS Spearman rho from evaluate.py (SAE supervised method)")
P.add_argument("--sae-spearman-all", type=float, default=None,
               help="All-years Spearman rho from evaluate.py")
P.add_argument("--sae-lifts-oos", type=str, default=None,
               help="Comma-separated OOS lift values for 0.5,1,2,5,10 pct")
P.add_argument("--sae-lifts-all", type=str, default=None,
               help="Comma-separated all-years lift values for 0.5,1,2,5,10 pct")
P.add_argument("--parent-spearman-oos", type=float, default=None,
               help="OOS Spearman rho for parent paper PCA cosine")
P.add_argument("--parent-spearman-all", type=float, default=None,
               help="All-years Spearman rho for parent paper PCA cosine")
P.add_argument("--parent-lifts-oos", type=str, default=None,
               help="Comma-separated OOS lift values for parent paper")
P.add_argument("--parent-lifts-all", type=str, default=None,
               help="Comma-separated all-years lift values for parent paper")
args = P.parse_args()

PCTS = [0.5, 1.0, 2.0, 5.0, 10.0]
BOLD = "\033[1m"
RESET = "\033[0m"

def _fix_separators(table_str):
    """Fix SEPARATING_LINE rendering in simple_outline format."""
    lines = table_str.split('\n')
    sep = None
    for line in lines:
        if line.lstrip().startswith('├') and '┼' in line:
            sep = line
            break
    if sep is None:
        return table_str
    fixed = []
    for line in lines:
        stripped = line.replace('│', '').replace(' ', '')
        if (stripped == '' and '│' in line
                and not line.lstrip().startswith('┌')
                and not line.lstrip().startswith('└')
                and not line.lstrip().startswith('├')):
            fixed.append(sep)
        else:
            fixed.append(line)
    return '\n'.join(fixed)

def _parse_lifts(csv_str):
    if csv_str is None:
        return None
    vals = [float(x.strip()) for x in csv_str.split(",")]
    assert len(vals) == len(PCTS), f"Expected {len(PCTS)} lift values, got {len(vals)}"
    return dict(zip(PCTS, vals))

sae_lifts_oos = _parse_lifts(args.sae_lifts_oos)
sae_lifts_all = _parse_lifts(args.sae_lifts_all)
parent_lifts_oos = _parse_lifts(args.parent_lifts_oos)
parent_lifts_all = _parse_lifts(args.parent_lifts_all)

# ---- Load features ----
df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

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

# Precompute norms on CPU
norms = np.linalg.norm(feat_matrix, axis=1).astype(np.float32)
norms = np.clip(norms, 1e-10, None)

# ---- Load pairs ----
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

# ==================================================================
# Compute dot products: double-chunked over features and pairs.
# ==================================================================

idx1_gpu = cp.asarray(idx1)
idx2_gpu = cp.asarray(idx2)
dot_sims_gpu = cp.zeros(n_pairs, dtype=cp.float32)

feat_chunk = args.feat_chunk
pair_batch = args.pair_batch
n_feat_chunks = (feat_dim + feat_chunk - 1) // feat_chunk
n_pair_batches = (n_pairs + pair_batch - 1) // pair_batch

for fi in range(n_feat_chunks):
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

# Cosine = dot / (norm1 * norm2)
cos_sims = dot_sims / (norms[idx1] * norms[idx2])

del idx1_gpu, idx2_gpu
cp.get_default_memory_pool().free_all_blocks()

# ---- Evaluate ----
test_mask = pairs_merged["year"].isin(test_years).values
test_corrs = correlations[test_mask]

# OOS metrics
test_dot = dot_sims[test_mask]
test_cos = cos_sims[test_mask]
test_sorted_dot = np.argsort(test_dot)[::-1]
test_sorted_cos = np.argsort(test_cos)[::-1]
rho_dot_oos, _ = spearmanr(test_dot, test_corrs)
rho_cos_oos, _ = spearmanr(test_cos, test_corrs)

# All-years metrics
all_sorted_dot = np.argsort(dot_sims)[::-1]
all_sorted_cos = np.argsort(cos_sims)[::-1]
rho_dot_all, _ = spearmanr(dot_sims, correlations)
rho_cos_all, _ = spearmanr(cos_sims, correlations)

# ---- Build report tables ----
def _fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"

def _bold(s):
    return f"{BOLD}{s}{RESET}"

def _build_table(cos_rho, dot_rho, sae_rho, parent_rho,
                 cos_sorted, dot_sorted, sae_lifts_dict, parent_lifts_dict,
                 ref_corrs, pop_mean):
    """Build a consolidated results table with bolded best values."""
    # Collect all lift values per cutoff to determine maxima
    all_lifts = {}  # {pct: [(method_idx, value), ...]}
    for pct in PCTS:
        entries = []
        n_top = max(1, int(len(ref_corrs) * pct / 100.0))
        entries.append((0, ref_corrs[cos_sorted[:n_top]].mean()))
        entries.append((1, ref_corrs[dot_sorted[:n_top]].mean()))
        if sae_lifts_dict:
            entries.append((2, sae_lifts_dict[pct]))
        if parent_lifts_dict:
            entries.append((3, parent_lifts_dict[pct]))
        all_lifts[pct] = entries

    best_lift = {}
    for pct, entries in all_lifts.items():
        best_lift[pct] = max(v for _, v in entries)

    rhos = [cos_rho, dot_rho, sae_rho, parent_rho]
    valid_rhos = [r for r in rhos if r is not None]
    best_rho = max(valid_rhos) if valid_rhos else None

    def _cell(val, is_best):
        s = _fmt(val)
        return _bold(s) if (is_best and val is not None) else s

    rows = []
    # Unsupervised SAE cosine
    for i, pct in enumerate(PCTS):
        n_top = max(1, int(len(ref_corrs) * pct / 100.0))
        lift_val = ref_corrs[cos_sorted[:n_top]].mean()
        rows.append([
            "Unsupervised SAE" if i == 0 else "(cosine)" if i == 1 else "",
            _cell(cos_rho, cos_rho == best_rho) if i == 0 else "",
            f"top {pct:.1f}%",
            _cell(lift_val, abs(lift_val - best_lift[pct]) < 1e-8),
        ])
    rows.append(SEPARATING_LINE)
    # Unsupervised SAE dot product
    for i, pct in enumerate(PCTS):
        n_top = max(1, int(len(ref_corrs) * pct / 100.0))
        lift_val = ref_corrs[dot_sorted[:n_top]].mean()
        rows.append([
            "Unsupervised SAE" if i == 0 else "(dot product)" if i == 1 else "",
            _cell(dot_rho, dot_rho == best_rho) if i == 0 else "",
            f"top {pct:.1f}%",
            _cell(lift_val, abs(lift_val - best_lift[pct]) < 1e-8),
        ])
    rows.append(SEPARATING_LINE)
    # New method (supervised selection)
    for i, pct in enumerate(PCTS):
        name = ["New method", "(Supervised", "selection)", "", ""][i]
        rho_cell = _cell(sae_rho, sae_rho == best_rho) if i == 0 else ""
        lift_val = sae_lifts_dict[pct] if sae_lifts_dict else None
        is_best = (lift_val is not None and abs(lift_val - best_lift[pct]) < 1e-8)
        rows.append([name, rho_cell, f"top {pct:.1f}%", _cell(lift_val, is_best)])
    rows.append(SEPARATING_LINE)
    # Parent paper (unsupervised PCA)
    for i, pct in enumerate(PCTS):
        name = ["Parent paper", "(Unsupervised", "PCA)", "", ""][i]
        rho_cell = _cell(parent_rho, parent_rho == best_rho) if i == 0 else ""
        lift_val = parent_lifts_dict[pct] if parent_lifts_dict else None
        is_best = (lift_val is not None and abs(lift_val - best_lift[pct]) < 1e-8)
        rows.append([name, rho_cell, f"top {pct:.1f}%", _cell(lift_val, is_best)])
    rows.append(SEPARATING_LINE)
    # Population mean
    rows.append(["Population", "", "", _fmt(pop_mean)])
    rows.append(["mean", "", "", ""])
    return _fix_separators(tabulate(rows,
                    headers=["Method", "Spearman rho", "Cutoff", "Mean correlation"],
                    tablefmt="simple_outline"))

# OOS table
oos_pop_mean = test_corrs.mean()
print(f"\nUnsupervised SAE Results (OOS):")
print(_build_table(
    cos_rho=rho_cos_oos, dot_rho=rho_dot_oos,
    sae_rho=args.sae_spearman_oos, parent_rho=args.parent_spearman_oos,
    cos_sorted=test_sorted_cos, dot_sorted=test_sorted_dot,
    sae_lifts_dict=sae_lifts_oos, parent_lifts_dict=parent_lifts_oos,
    ref_corrs=test_corrs, pop_mean=oos_pop_mean,
))

# All-years table
all_pop_mean = correlations.mean()
print(f"\nUnsupervised SAE Results (All years):")
print(_build_table(
    cos_rho=rho_cos_all, dot_rho=rho_dot_all,
    sae_rho=args.sae_spearman_all, parent_rho=args.parent_spearman_all,
    cos_sorted=all_sorted_cos, dot_sorted=all_sorted_dot,
    sae_lifts_dict=sae_lifts_all, parent_lifts_dict=parent_lifts_all,
    ref_corrs=correlations, pop_mean=all_pop_mean,
))
