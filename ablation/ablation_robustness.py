#!/usr/bin/env python3


"""initial setup imports"""
import argparse
import numpy as np
import pandas as pd
import cupy as cp
import joblib
from datasets import load_dataset
from scipy.stats import spearmanr
from tabulate import tabulate, SEPARATING_LINE
import gc


"""helper function to unwrap features from the dataframe, ensuring they are in the correct format for further processing."""
def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()

"""Argument parsing to specify input files, model loading, and parameters for the analysis."""
P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--top-k", type=int, default=1250)
P.add_argument("--pair-batch", type=int, default=500_000)
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

"""The script begins by loading the features from a pickle file and the covariate dataset using the Hugging Face datasets library. 
The features are unwrapped and converted to a suitable format, while the covariate dataset is merged with the features based on a common index.
 The script ensures that any rows with missing SIC codes are dropped,
 and the year column is properly formatted as integers."""
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

"""The features are extracted into a matrix, 
and any rows containing NaN or infinite values are removed to ensure the integrity of the data for subsequent analysis."""
feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

"""The pre-trained model is loaded using joblib, and the scores for each feature dimension are extracted.
 The feature dimensions are ranked based on their scores, 
 and the top-k dimensions with positive scores are selected for further analysis. 
 The selected features are then weighted by their corresponding scores to create a new feature representation that emphasizes the most relevant dimensions according to the model's scoring."""
saved = joblib.load(args.load_model)
scores = saved["scores"]
ranked = np.argsort(scores)[::-1]
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)


selected_features = feat_matrix[:, selected].copy().astype(np.float32)
weights = scores[selected].astype(np.float32)
selected_features *= weights[np.newaxis, :]

"""features are normalized and selected to do cosine similarity"""
feat_norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10).astype(np.float32)

del feat_matrix; gc.collect()

"""The original pairs dataset is loaded, 
and any rows with missing correlation values are dropped. 
The year and company identifiers are properly formatted, 
and the dataset is merged with the feature index to associate each company in the pairs with its corresponding feature index. 
This allows for efficient retrieval of the selected features for each company when computing the similarity scores for the pairs."""
pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)

"""train test split based on years"""
all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
test_years = set(all_years[split_idx:])
test_years_no2020 = test_years - {2020}

"""The company identifiers from the features dataframe are extracted, 
and a new dataframe is created to map each company and year to its corresponding feature index.
 This mapping is then merged with the pairs dataset to associate each company in the pairs with its feature index, 
 allowing for efficient retrieval of the selected features for each company when computing the similarity scores for the pairs."""
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


sel_gpu = cp.asarray(selected_features)
idx1_gpu = cp.asarray(idx1)
idx2_gpu = cp.asarray(idx2)
sims = np.empty(n_pairs, dtype=np.float32)

"""The similarity scores for all pairs are computed using the selected features. 
The computation is done in batches to manage memory usage, 
where the dot product of the selected features for the two companies in each pair is calculated to obtain the similarity score. 
The scores are stored in a NumPy array, and the GPU memory is freed after the computation to ensure efficient resource management"""
pair_batch = args.pair_batch
for s in range(0, n_pairs, pair_batch):
    e = min(s + pair_batch, n_pairs)
    a = sel_gpu[idx1_gpu[s:e]]
    b = sel_gpu[idx2_gpu[s:e]]
    sims[s:e] = cp.asnumpy((a * b).sum(axis=1))
    del a, b

del sel_gpu, idx1_gpu, idx2_gpu
cp.get_default_memory_pool().free_all_blocks()

"""this makes the results tables"""
BOLD = "\033[1m"
RESET = "\033[0m"


def _fix_separators(table_str):
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
        stripped = line.replace('│', '').replace(' ', '').replace('\x01', '')
        if (stripped == '' and '│' in line
                and not line.lstrip().startswith('┌')
                and not line.lstrip().startswith('└')
                and not line.lstrip().startswith('├')):
            fixed.append(sep)
        else:
            fixed.append(line)
    return '\n'.join(fixed)


def _fmt(val):
    return f"{val:.4f}"


def _bold(s):
    return f"{BOLD}{s}{RESET}"


def _cell(val, is_best):
    s = _fmt(val)
    return _bold(s) if is_best else s


test_mask_full = pairs_merged["year"].isin(test_years).values
test_mask_no2020 = pairs_merged["year"].isin(test_years_no2020).values
mask_2020 = pairs_merged["year"].eq(2020).values

test_sims = sims[test_mask_full]
test_corrs = correlations[test_mask_full]
rho_baseline, _ = spearmanr(test_sims, test_corrs)
sorted_baseline = np.argsort(test_sims)[::-1]

sims_no2020 = sims[test_mask_no2020]
corrs_no2020 = correlations[test_mask_no2020]
rho_no2020, _ = spearmanr(sims_no2020, corrs_no2020)
sorted_no2020 = np.argsort(sims_no2020)[::-1]

norm_products = feat_norms[idx1] * feat_norms[idx2]
test_norm_prods = norm_products[test_mask_full]
corr_sim_norm = np.corrcoef(sims, norm_products)[0, 1]
slope = np.cov(test_norm_prods, test_sims)[0, 1] / np.var(test_norm_prods)
intercept = test_sims.mean() - slope * test_norm_prods.mean()
residuals_test = test_sims - (slope * test_norm_prods + intercept)
r_squared = 1 - np.var(residuals_test) / np.var(test_sims)
rho_resid, _ = spearmanr(residuals_test, test_corrs)
sorted_resid = np.argsort(residuals_test)[::-1]

pop_mean_full = test_corrs.mean()
pop_mean_no2020 = corrs_no2020.mean()

n_test = test_mask_full.sum()
n_2020 = mask_2020.sum()

conditions = [
    ("OOS baseline", rho_baseline, sorted_baseline, test_corrs),
    ("OOS excl. 2020", rho_no2020, sorted_no2020, corrs_no2020),
    ("Norm-residualized", rho_resid, sorted_resid, test_corrs),
]

lifts = {}
for name, _rho, srt, corrs in conditions:
    lifts[name] = {}
    for pct in PCTS:
        n_top = max(1, int(len(corrs) * pct / 100.0))
        lifts[name][pct] = corrs[srt[:n_top]].mean()

all_rhos = [c[1] for c in conditions]
best_rho = max(all_rhos)
best_lift = {pct: max(lifts[n][pct] for n, _, _, _ in conditions) for pct in PCTS}

rows = []
for name, rho, _srt, _corrs in conditions:
    for i, pct in enumerate(PCTS):
        lift_val = lifts[name][pct]
        rows.append([
            name if i == 0 else "",
            _cell(rho, abs(rho - best_rho) < 1e-8) if i == 0 else "",
            f"top {pct:.1f}%",
            _cell(lift_val, abs(lift_val - best_lift[pct]) < 1e-8),
        ])
    rows.append(SEPARATING_LINE)

rows.append(["Population mean", "", "", _fmt(pop_mean_full)])
rows.append(["Population mean", "", "", _fmt(pop_mean_no2020)])
rows.append(["(excl. 2020)", "", "", ""])

table_main = _fix_separators(tabulate(rows,
                      headers=["Method", "Spearman rho", "Cutoff", "Mean correlation"],
                      tablefmt="simple_outline"))

year_data = []
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
    year_data.append((year, ymask.sum(), yr, top1_mean, yc.mean()))

best_year_rho = max(d[2] for d in year_data)
best_year_top1 = max(d[3] for d in year_data)

year_rows = []
for year, n_pairs_y, yr, top1, popc in year_data:
    year_rows.append([
        str(year),
        f"{n_pairs_y:,d}",
        _cell(yr, abs(yr - best_year_rho) < 1e-8),
        _cell(top1, abs(top1 - best_year_top1) < 1e-8),
        _fmt(popc),
    ])

table_years = tabulate(year_rows,
                       headers=["Year", "Pairs", "Spearman rho", "Top 1%", "Pop corr"],
                       tablefmt="simple_outline",
                       colalign=("right", "right", "right", "right", "right"))

print(table_main)
print(f"\nBy Year (OOS):")
print(table_years)
print(f"\nOther Stats:\n2020 Population MC: {correlations[mask_2020].mean():.4f}")
print(f"Corr(Similarity, Norm_Product): {corr_sim_norm:.4f}")
print(f"R^2 (Norm vs. Similarity): {r_squared:.4f}")