#!/usr/bin/env python3
"""
Unified precision-at-k evaluation: our approach, parent paper, SIC baseline.

Usage (Colab):
    !python unified_eval.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl \
        --load-model /content/drive/MyDrive/company_similarity_sae/data/llama_selection_model.pkl \
        --top-k 1000
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from scipy.stats import spearmanr
from tabulate import tabulate
import gc


def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--top-k", type=int, default=1000)
P.add_argument("--score-weight", action="store_true", default=True)
P.add_argument("--no-score-weight", action="store_false", dest="score_weight")
P.add_argument("--norm-alpha", type=float, default=0.0,
               help="Norm exponent: 0=dot product, 1=cosine similarity")
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

# Load shared data

raw = load_dataset(args.original_pairs_ds)["train"].to_pandas()

ds = raw.dropna(subset=["correlation", "cosine_similarity"]).copy()
ds["year"] = ds["year"].astype(int)
ds["Company1"] = ds["Company1"].astype(str)
ds["Company2"] = ds["Company2"].astype(str)

pairs_df = raw.dropna(subset=["correlation"]).copy()
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)
del raw

all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
test_years = set(all_years[split_idx:])
pop_corr = pairs_df["correlation"].mean()

df_c = load_dataset(args.cov_ds)["train"].to_pandas()
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

# A: Parent paper (PCA 4000-dim cosine similarity)

cos_sims_A = ds["cosine_similarity"].values.astype(np.float32)
corrs_A = ds["correlation"].values.astype(np.float32)
rho_A, pval_A = spearmanr(cos_sims_A, corrs_A)

test_mask_A = ds["year"].isin(test_years).values
test_sims_A = cos_sims_A[test_mask_A]
test_corrs_A = corrs_A[test_mask_A]
test_sorted_A = np.argsort(test_sims_A)[::-1]

prec_A = {}
for pct in PCTS:
    n_top = max(1, int(len(test_sims_A) * pct / 100.0))
    prec_A[pct] = test_corrs_A[test_sorted_A[:n_top]].mean()

# B: Our approach (supervised SAE feature selection)

df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)
del df_f

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

selected_features = feat_matrix[:, selected].copy()
if args.score_weight:
    weights = scores[selected].astype(np.float32)
    selected_features *= weights[np.newaxis, :]
norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10)

company_ids = df["__index_level_0__"].values
feat_idx_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

pairs_B = pairs_df.merge(
    feat_idx_df.rename(columns={"__index_level_0__": "Company1", "feat_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_B = pairs_B.merge(
    feat_idx_df.rename(columns={"__index_level_0__": "Company2", "feat_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)

idx1 = pairs_B["idx1"].values
idx2 = pairs_B["idx2"].values
corrs_B = pairs_B["correlation"].values.astype(np.float32)

cos_sims_B = np.empty(len(pairs_B), dtype=np.float32)
batch = 500_000
alpha = args.norm_alpha
for s in range(0, len(pairs_B), batch):
    e = min(s + batch, len(pairs_B))
    i1, i2 = idx1[s:e], idx2[s:e]
    dot = (selected_features[i1] * selected_features[i2]).sum(1)
    if alpha > 0:
        cos_sims_B[s:e] = dot / ((norms[i1] * norms[i2]) ** alpha)
    else:
        cos_sims_B[s:e] = dot

rho_B, pval_B = spearmanr(cos_sims_B, corrs_B)

test_mask_B = pairs_B["year"].isin(test_years).values
test_sims_B = cos_sims_B[test_mask_B]
test_corrs_B = corrs_B[test_mask_B]
test_sorted_B = np.argsort(test_sims_B)[::-1]

prec_B = {}
for pct in PCTS:
    n_top = max(1, int(len(test_sims_B) * pct / 100.0))
    prec_B[pct] = test_corrs_B[test_sorted_B[:n_top]].mean()

del feat_matrix, selected_features
gc.collect()

# C: SIC industry code baseline

sic_info = df_c[["__index_level_0__", "year", "sic_code"]].dropna(subset=["sic_code"]).copy()
sic_dedup = sic_info.drop_duplicates(subset=["__index_level_0__", "year"], keep="last")

ds_sic = ds.merge(
    sic_dedup.rename(columns={"__index_level_0__": "Company1", "sic_code": "sic1"}),
    on=["Company1", "year"], how="inner",
).merge(
    sic_dedup.rename(columns={"__index_level_0__": "Company2", "sic_code": "sic2"}),
    on=["Company2", "year"], how="inner",
)

same_sic = ds_sic["sic1"] == ds_sic["sic2"]
test_mask_sic = ds_sic["year"].isin(test_years)
ds_sic_test = ds_sic[test_mask_sic]
same_sic_test = ds_sic_test["sic1"] == ds_sic_test["sic2"]

n_same_test = int(same_sic_test.sum())
n_total_test_sic = len(ds_sic_test)
pct_same_test = 100.0 * n_same_test / n_total_test_sic
sic_corr_test = ds_sic_test.loc[same_sic_test, "correlation"].mean()

# Results

BOLD = "\033[1m"
RESET = "\033[0m"
yr_min, yr_max = min(test_years), max(test_years)

spearman_table = [
    ["New approach (k={}, α={})".format(n_selected, args.norm_alpha), f"{rho_B:.4f}", f"{pval_B:.2e}"],
    ["Parent paper (PCA 4000-dim)", f"{rho_A:.4f}", f"{pval_A:.2e}"],
]

prec_table = []
for pct in PCTS:
    sic_val = f"{sic_corr_test:.4f}" if pct == 1.0 else ""
    row = [f"top {pct:.1f}%", f"{prec_B[pct]:.4f}", f"{prec_A[pct]:.4f}", sic_val]
    if pct == 1.0:
        row = [f"{BOLD}{c}{RESET}" for c in row]
    prec_table.append(row)

print()
print("Spearman rank correlation (all years)")
print(tabulate(
    spearman_table,
    headers=["Approach", "Spearman rho", "p-value"],
    tablefmt="simple_outline",
))

print()
print(f"Precision-at-k: mean return correlation, OOS {yr_min}-{yr_max}")
print(tabulate(
    prec_table,
    headers=["Cutoff", "New approach", "Parent paper", "SIC baseline"],
    tablefmt="simple_outline",
))

print()
print(f"Population mean return correlation: {pop_corr:.4f}")
print()
print(f"New approach: supervised selection, k={n_selected}, "
      f"score-weighted={'yes' if args.score_weight else 'no'}, "
      f"α={args.norm_alpha}; "
      f"{len(test_sims_B):,d} OOS pairs")
print(f"Parent paper: PCA 4000-dim cosine similarity; "
      f"{len(test_sims_A):,d} OOS pairs")
print(f"SIC baseline: {n_same_test:,d} same-code pairs out of "
      f"{n_total_test_sic:,d} ({pct_same_test:.2f}%), shown at top 1.0%")