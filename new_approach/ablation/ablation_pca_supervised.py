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
P.add_argument("--sae-spearman-oos", type=float, default=None,
               help="OOS Spearman rho from evaluate.py (SAE supervised method)")
P.add_argument("--sae-spearman-all", type=float, default=None,
               help="All-years Spearman rho from evaluate.py")
P.add_argument("--sae-lifts-oos", type=str, default=None,
               help="Comma-separated OOS lift values for 0.5,1,2,5,10 pct")
P.add_argument("--sae-lifts-all", type=str, default=None,
               help="Comma-separated all-years lift values for 0.5,1,2,5,10 pct")
args = P.parse_args()

PCTS = [0.5, 1.0, 2.0, 5.0, 10.0]

def _parse_lifts(csv_str):
    if csv_str is None:
        return None
    vals = [float(x.strip()) for x in csv_str.split(",")]
    assert len(vals) == len(PCTS), f"Expected {len(PCTS)} lift values, got {len(vals)}"
    return dict(zip(PCTS, vals))

sae_lifts_oos = _parse_lifts(args.sae_lifts_oos)
sae_lifts_all = _parse_lifts(args.sae_lifts_all)

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

# ---- Fit PCA (globally, matching parent paper) ----
pca = PCA(n_components=args.pca_dims, svd_solver="randomized", random_state=42)
pca_features = pca.fit_transform(feat_matrix).astype(np.float32)

del feat_matrix; gc.collect()

# ---- Load pairs ----
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

# ---- Score each PCA dimension ----
# For each dim j: Pearson corr between (pca[i1,j] * pca[i2,j]) and return_corr
scores = np.zeros(args.pca_dims, dtype=np.float64)

corr_train_demean = corr_train - corr_train.mean()
corr_train_std = corr_train.std()

for j in range(args.pca_dims):
    products = pca_features[idx1_train, j] * pca_features[idx2_train, j]
    prod_demean = products - products.mean()
    prod_std = prod_demean.std()
    if prod_std > 0:
        scores[j] = (prod_demean * corr_train_demean).mean() / (prod_std * corr_train_std)

n_positive = (scores > 0).sum()

# ---- Select top-k ----
ranked = np.argsort(scores)[::-1]
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)

if n_selected == 0:
    print("ERROR: No PCA dimensions scored positively. Exiting.")
    exit(1)

# ---- Compute similarities (score-weighted dot product, alpha=0) ----
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

# ---- Unsupervised PCA cosine (matching parent paper) ----
norms_pca = np.linalg.norm(pca_features, axis=1).clip(min=1e-10)

cos_unsup = np.empty(len(pairs_merged), dtype=np.float32)
for s in range(0, len(pairs_merged), batch):
    e = min(s + batch, len(pairs_merged))
    i1, i2 = idx1[s:e], idx2[s:e]
    dot = (pca_features[i1] * pca_features[i2]).sum(1)
    cos_unsup[s:e] = dot / (norms_pca[i1] * norms_pca[i2])

rho_unsup_all, _ = spearmanr(cos_unsup, correlations)
test_cos_unsup = cos_unsup[test_mask_eval]
test_sorted_unsup = np.argsort(test_cos_unsup)[::-1]
rho_unsup_oos, _ = spearmanr(test_cos_unsup, test_corrs)

# ---- Compute all-years lifts (supervised PCA) ----
all_sorted_sup = np.argsort(sims)[::-1]

# ---- Compute all-years lifts (unsupervised PCA) ----
all_sorted_unsup = np.argsort(cos_unsup)[::-1]

# ---- Build report tables ----
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

def _fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"

def _bold(s):
    return f"{BOLD}{s}{RESET}"

def _build_table(sup_rho, unsup_rho, sae_rho,
                 sup_sorted, unsup_sorted, sae_lifts_dict,
                 ref_corrs, pop_mean):
    """Build a consolidated results table with bolded best values."""
    # Collect all lift values per cutoff to determine maxima
    all_lifts = {}
    for pct in PCTS:
        entries = []
        n_top = max(1, int(len(ref_corrs) * pct / 100.0))
        entries.append(ref_corrs[sup_sorted[:n_top]].mean())
        if sae_lifts_dict:
            entries.append(sae_lifts_dict[pct])
        entries.append(ref_corrs[unsup_sorted[:n_top]].mean())
        all_lifts[pct] = entries

    best_lift = {pct: max(vals) for pct, vals in all_lifts.items()}

    rhos = [sup_rho, sae_rho, unsup_rho]
    valid_rhos = [r for r in rhos if r is not None]
    best_rho = max(valid_rhos) if valid_rhos else None

    def _cell(val, is_best):
        s = _fmt(val)
        return _bold(s) if (is_best and val is not None) else s

    rows = []
    # Supervised PCA rows
    for i, pct in enumerate(PCTS):
        n_top = max(1, int(len(ref_corrs) * pct / 100.0))
        top_mean = ref_corrs[sup_sorted[:n_top]].mean()
        rows.append([
            "Supervised PCA" if i == 0 else "",
            _cell(sup_rho, sup_rho == best_rho) if i == 0 else "",
            f"top {pct:.1f}%",
            _cell(top_mean, abs(top_mean - best_lift[pct]) < 1e-8),
        ])
    rows.append(SEPARATING_LINE)
    # SAE (new method) rows
    for i, pct in enumerate(PCTS):
        name = ["New method", "(Supervised", "selection)", "", ""][i]
        rho_cell = _cell(sae_rho, sae_rho == best_rho) if i == 0 else ""
        lift_val = sae_lifts_dict[pct] if sae_lifts_dict else None
        is_best = (lift_val is not None and abs(lift_val - best_lift[pct]) < 1e-8)
        rows.append([name, rho_cell, f"top {pct:.1f}%", _cell(lift_val, is_best)])
    rows.append(SEPARATING_LINE)
    # Parent paper (unsupervised PCA) rows
    for i, pct in enumerate(PCTS):
        name = ["Parent paper", "(Unsupervised", "PCA)", "", ""][i]
        rho_cell = _cell(unsup_rho, unsup_rho == best_rho) if i == 0 else ""
        n_top = max(1, int(len(ref_corrs) * pct / 100.0))
        top_mean = ref_corrs[unsup_sorted[:n_top]].mean()
        rows.append([name, rho_cell, f"top {pct:.1f}%",
                     _cell(top_mean, abs(top_mean - best_lift[pct]) < 1e-8)])
    rows.append(SEPARATING_LINE)
    # Population mean
    rows.append(["Population", "", "", _fmt(pop_mean)])
    rows.append(["mean", "", "", ""])
    return _fix_separators(tabulate(rows,
                    headers=["Method", "Spearman rho", "Cutoff", "Mean correlation"],
                    tablefmt="simple_outline"))

# OOS table
oos_pop_mean = test_corrs.mean()
print(f"\nSupervised PCA Results (OOS {min(test_years)}-{max(test_years)}, "
      f"{len(test_corrs):,d} pairs):")
print(_build_table(
    sup_rho=rho_test, unsup_rho=rho_unsup_oos, sae_rho=args.sae_spearman_oos,
    sup_sorted=test_sorted, unsup_sorted=test_sorted_unsup,
    sae_lifts_dict=sae_lifts_oos,
    ref_corrs=test_corrs, pop_mean=oos_pop_mean,
))

# All-years table
all_pop_mean = correlations.mean()
print(f"\nSupervised PCA Results (All years, {len(correlations):,d} pairs):")
print(_build_table(
    sup_rho=rho_all, unsup_rho=rho_unsup_all, sae_rho=args.sae_spearman_all,
    sup_sorted=all_sorted_sup, unsup_sorted=all_sorted_unsup,
    sae_lifts_dict=sae_lifts_all,
    ref_corrs=correlations, pop_mean=all_pop_mean,
))
