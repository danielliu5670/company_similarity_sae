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
P.add_argument("--sbert-pkl", default=None,
               help="SBERT embeddings pkl (Vamvourellis et al. baseline)")
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

# D: Vamvourellis et al. (SBERT embedding cosine similarity)

rho_D = pval_D = None
prec_D = {}
n_test_D = 0

if args.sbert_pkl:
    df_sbert = pd.read_pickle(args.sbert_pkl)
    df_sbert["__index_level_0__"] = df_sbert["__index_level_0__"].astype(str)

    df_sb = pd.merge(df_sbert, df_c, on="__index_level_0__", how="inner")
    df_sb = df_sb.dropna(subset=["sic_code"])
    df_sb["year"] = df_sb["year"].astype(int)

    sbert_matrix = np.vstack(df_sb["sbert_embedding"].values).astype(np.float32)
    sbert_norms = np.linalg.norm(sbert_matrix, axis=1).clip(min=1e-10)

    sbert_idx_df = pd.DataFrame({
        "__index_level_0__": df_sb["__index_level_0__"].values,
        "year": df_sb["year"].values,
        "sbert_idx": np.arange(len(df_sb)),
    })

    pairs_D = pairs_df.merge(
        sbert_idx_df.rename(columns={
            "__index_level_0__": "Company1", "sbert_idx": "sidx1"}),
        on=["Company1", "year"], how="inner",
    )
    pairs_D = pairs_D.merge(
        sbert_idx_df.rename(columns={
            "__index_level_0__": "Company2", "sbert_idx": "sidx2"}),
        on=["Company2", "year"], how="inner",
    )

    sidx1 = pairs_D["sidx1"].values
    sidx2 = pairs_D["sidx2"].values
    corrs_D = pairs_D["correlation"].values.astype(np.float32)

    cos_sims_D = np.empty(len(pairs_D), dtype=np.float32)
    for s in range(0, len(pairs_D), 500_000):
        e = min(s + 500_000, len(pairs_D))
        i1, i2 = sidx1[s:e], sidx2[s:e]
        dot = (sbert_matrix[i1] * sbert_matrix[i2]).sum(1)
        cos_sims_D[s:e] = dot / (sbert_norms[i1] * sbert_norms[i2])

    rho_D, pval_D = spearmanr(cos_sims_D, corrs_D)

    test_mask_D = pairs_D["year"].isin(test_years).values
    test_sims_D = cos_sims_D[test_mask_D]
    test_corrs_D = corrs_D[test_mask_D]
    test_sorted_D = np.argsort(test_sims_D)[::-1]
    n_test_D = len(test_sims_D)

    for pct in PCTS:
        n_top = max(1, int(len(test_sims_D) * pct / 100.0))
        prec_D[pct] = test_corrs_D[test_sorted_D[:n_top]].mean()

    del sbert_matrix, df_sbert, df_sb
    gc.collect()

# Results

BOLD = "\033[1m"
RESET = "\033[0m"
yr_min, yr_max = min(test_years), max(test_years)

spearman_entries = [
    ("New approach (k={}, \u03b1={})".format(n_selected, args.norm_alpha), rho_B, pval_B),
    ("Parent paper (PCA 4000-dim)", rho_A, pval_A),
]
if rho_D is not None:
    spearman_entries.append(("Vamvourellis (SBERT cosine)", rho_D, pval_D))
best_rho = max(e[1] for e in spearman_entries)

spearman_table = []
for name, rho, pval in spearman_entries:
    rho_s = f"{rho:.4f}"
    if abs(rho - best_rho) < 1e-8:
        rho_s = f"{BOLD}{rho_s}{RESET}"
    spearman_table.append([name, rho_s, f"{pval:.2e}"])

prec_headers = ["Cutoff", "New approach", "Parent paper"]
if prec_D:
    prec_headers.append("Vamvourellis")
prec_headers.append("SIC baseline")

def _b(v, best):
    s = f"{v:.4f}"
    return f"{BOLD}{s}{RESET}" if abs(v - best) < 1e-8 else s

prec_table = []
for pct in PCTS:
    sic_val = sic_corr_test if pct == 1.0 else None
    row_vals = [prec_B[pct], prec_A[pct]]
    if prec_D:
        row_vals.append(prec_D[pct])
    if sic_val is not None:
        row_vals.append(sic_val)
    best_val = max(row_vals)

    row = [f"top {pct:.1f}%", _b(prec_B[pct], best_val), _b(prec_A[pct], best_val)]
    if prec_D:
        row.append(_b(prec_D[pct], best_val))
    row.append(_b(sic_val, best_val) if sic_val is not None else "")
    prec_table.append(row)

print()
print("Spearman rank correlation (all years)")
print(tabulate(
    spearman_table,
    headers=["Approach", "Spearman rho", "p-value"],
    tablefmt="simple_outline",
))

print()
print(f"Mean return correlation, OOS {yr_min}-{yr_max}")
print(tabulate(
    prec_table,
    headers=prec_headers,
    tablefmt="simple_outline",
))

print()
print(f"Population mean return correlation: {pop_corr:.4f}")
print()
print(f"New approach: k={n_selected}, "
      f"score-weighted={'yes' if args.score_weight else 'no'}, "
      f"\u03b1={args.norm_alpha}; "
      f"{len(test_sims_B):,d} OOS pairs")
print(f"Parent paper: PCA 4000-dim cosine; "
      f"{len(test_sims_A):,d} OOS pairs")
if n_test_D > 0:
    print(f"Vamvourellis: SBERT all-mpnet-base-v2 cosine; "
          f"{n_test_D:,d} OOS pairs")
print(f"SIC: {n_same_test:,d} same-code / "
      f"{n_total_test_sic:,d} ({pct_same_test:.1f}%)")