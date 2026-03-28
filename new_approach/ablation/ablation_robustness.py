#!/usr/bin/env python3
"""
Robustness checks for the supervised SAE approach:
  1. Exclude 2020 from OOS evaluation (COVID-19 correlation spike)
  2. Norm residualization (separate content signal from magnitude signal)

Usage (Colab):
    !python ablation_robustness.py \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl \
        --load-model /content/drive/MyDrive/company_similarity_sae/data/llama_selection_model.pkl \
        --top-k 1250
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
P.add_argument("--top-k", type=int, default=1250)
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

# ---- Load data (same as evaluate.py) ----
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

# ---- Load model, select features ----
saved = joblib.load(args.load_model)
scores = saved["scores"]
ranked = np.argsort(scores)[::-1]
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)
print(f"  Selected {n_selected} features")

selected_features = feat_matrix[:, selected].copy()
weights = scores[selected].astype(np.float32)
selected_features *= weights[np.newaxis, :]

# Norms of the score-weighted selected features (for residualization)
feat_norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10).astype(np.float32)

del feat_matrix; gc.collect()

# ---- Load pairs ----
print("Loading pairs...")
pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)

# ---- Temporal split ----
all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
test_years = set(all_years[split_idx:])
test_years_no2020 = test_years - {2020}

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

# ---- Compute similarities (dot product, alpha=0) ----
print("Computing similarities...")
sims = np.empty(len(pairs_merged), dtype=np.float32)
batch = 500_000
for s in range(0, len(pairs_merged), batch):
    e = min(s + batch, len(pairs_merged))
    i1, i2 = idx1[s:e], idx2[s:e]
    sims[s:e] = (selected_features[i1] * selected_features[i2]).sum(1)

# ---- Evaluation helper ----
def evaluate(sim_values, corr_values, label, year_mask=None):
    """Evaluate on a subset defined by year_mask (boolean array over sim_values)."""
    if year_mask is not None:
        s = sim_values[year_mask]
        c = corr_values[year_mask]
    else:
        s = sim_values
        c = corr_values

    rho, pval = spearmanr(s, c)
    sorted_idx = np.argsort(s)[::-1]

    print(f"\n--- {label} ({len(s):,d} pairs) ---")
    print(f"Spearman rho: {rho:.4f} (p={pval:.2e})")
    rows = []
    for pct in PCTS:
        n_top = max(1, int(len(s) * pct / 100.0))
        top_mean = c[sorted_idx[:n_top]].mean()
        rows.append([f"top {pct:.1f}%", f"{top_mean:.4f}"])
    print(tabulate(rows, headers=["Cutoff", "Mean return corr"],
                   tablefmt="simple_outline"))
    return rho

# ================================================================
# CHECK 1: Exclude 2020
# ================================================================
print(f"\n{'='*70}")
print("CHECK 1: EXCLUDE 2020 FROM OOS")
print(f"{'='*70}")

test_mask_full = pairs_merged["year"].isin(test_years).values
test_mask_no2020 = pairs_merged["year"].isin(test_years_no2020).values
mask_2020_only = pairs_merged["year"].eq(2020).values

n_2020 = mask_2020_only.sum()
n_test = test_mask_full.sum()
print(f"\n  2020 pairs: {n_2020:,d} out of {n_test:,d} OOS pairs "
      f"({100*n_2020/n_test:.1f}%)")
print(f"  Mean return corr in 2020 pairs: {correlations[mask_2020_only].mean():.4f}")
print(f"  Mean return corr in non-2020 OOS: {correlations[test_mask_no2020].mean():.4f}")

evaluate(sims, correlations, "OOS with 2020 (baseline)", test_mask_full)
evaluate(sims, correlations, "OOS without 2020", test_mask_no2020)

# Per-year breakdown
print("\n  Per-year OOS breakdown:")
print(f"  {'Year':>6s}  {'Pairs':>10s}  {'Spearman':>10s}  {'Top 1%':>10s}  {'Mean corr':>10s}")
for year in sorted(test_years):
    ymask = pairs_merged["year"].eq(year).values
    if ymask.sum() < 100:
        continue
    ys = sims[ymask]
    yc = correlations[ymask]
    yr, _ = spearmanr(ys, yc)
    ysorted = np.argsort(ys)[::-1]
    ntop = max(1, int(len(ys) * 1.0 / 100.0))
    top1_mean = yc[ysorted[:ntop]].mean()
    print(f"  {year:>6d}  {ymask.sum():>10,d}  {yr:>10.4f}  {top1_mean:>10.4f}  {yc.mean():>10.4f}")

# ================================================================
# CHECK 2: Norm residualization
# ================================================================
print(f"\n{'='*70}")
print("CHECK 2: NORM RESIDUALIZATION")
print(f"{'='*70}")

# For each pair, compute the product of both companies' norms
norm_products = feat_norms[idx1] * feat_norms[idx2]

print(f"\n  Correlation between dot product sim and norm product: "
      f"{np.corrcoef(sims, norm_products)[0,1]:.4f}")

# OLS: sims = a * norm_products + b + residual
# Using only test pairs for the regression (to avoid any train contamination)
test_sims = sims[test_mask_full]
test_corrs = correlations[test_mask_full]
test_norm_prods = norm_products[test_mask_full]

# Fit OLS on test set
X = test_norm_prods
Y = test_sims
slope = np.cov(X, Y)[0, 1] / np.var(X)
intercept = Y.mean() - slope * X.mean()
residuals_test = Y - (slope * X + intercept)

print(f"  OLS: sim = {slope:.6f} * norm_product + {intercept:.6f}")
print(f"  R^2 of norm product on similarity: "
      f"{1 - np.var(residuals_test)/np.var(test_sims):.4f}")

evaluate(test_sims, test_corrs, "OOS raw dot product (baseline)")
evaluate(residuals_test, test_corrs, "OOS after norm residualization")

# Also residualize on all pairs and evaluate
all_residuals = sims - (slope * norm_products + intercept)
evaluate(all_residuals, correlations, "All years after norm residualization",
         test_mask_full)

print(f"\n{'='*70}")
print("INTERPRETATION:")
print("  If top-1% lift remains strong after residualization,")
print("  the signal is content-driven, not magnitude-driven.")
print("  If it collapses, description length is a confound.")
print(f"{'='*70}")
