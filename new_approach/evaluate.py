#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import joblib
from datasets import load_dataset
from scipy.stats import spearmanr
from tabulate import tabulate
import gc

"""
This function essentially "peels back" all of the potential nesting locations 
that the features could be stored in. It then takes these features and 
returns them as a 1D array.
"""

def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1: # As you can see, it repeatedly "probes"
                                                 # the inner arrays of something until
                                                 # it reaches an actual element.
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten() # Then, it repeatedly flattens these elements
                                                   # and returns it.

"""
These are all the command-line arguments that can be passed into the file.
"""

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

PCTS = [0.5, 1.0, 2.0, 5.0, 10.0] # While not a command line argument, this is
                                  # the set of top-% values that will be assessed,
                                  # so change this if you would like.

"""
This section loads the data from the parent paper from HuggingFace. Two datasets
are created - ds, which requires both the return correlation and cosine similarity
between the two companies, and pairs_df, which only requires the correlation.
"""

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

"""
This section creates the training / testing split, specifically being 75/25 across
years. This is not done across rows to avoid data leakage and various other problems.

It also defines the population mean correlation (pop_corr).
"""

all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
test_years = set(all_years[split_idx:])
pop_corr = pairs_df["correlation"].mean()

"""
This section loads the metadata, also from HuggingFace.
"""

df_c = load_dataset(args.cov_ds)["train"].to_pandas()
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

"""
This section evaluates the correlation-at-k and Spearman rho for the original approach.
Specifically, it first extracts all of the cosine similarities from the dataset,
then calculates its Spearman rho correlation with the actual similarities. 

Then, for each of the top-% cutoffs specified above, it calculates the correlation-at-k.
"""

cos_sims_A = ds["cosine_similarity"].values.astype(np.float32)
corrs_A = ds["correlation"].values.astype(np.float32)

test_mask_A = ds["year"].isin(test_years).values
test_sims_A = cos_sims_A[test_mask_A]
test_corrs_A = corrs_A[test_mask_A]
test_sorted_A = np.argsort(test_sims_A)[::-1]

rho_A, pval_A = spearmanr(test_sims_A, test_corrs_A)

prec_A = {}
for pct in PCTS:
    n_top = max(1, int(len(test_sims_A) * pct / 100.0))
    prec_A[pct] = test_corrs_A[test_sorted_A[:n_top]].mean() # This section computes the correlation-at-k, which
                                                             # which is simply the average actual similarities 
                                                             # between the top-% companies that are ranked the highest
                                                             # in terms of the computed similarity.

"""
This section pulls the sparse autoencoder feature vectors from the input pickle file 
(see collect_features.py). It then, for each company, merges them with the metadata.
"""

df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"]) # If any of the companies does not have an SIC code,
                                    # it is dropped. This is because knowing a company's
                                    # SIC is necessary for the baseline.
df["year"] = df["year"].astype(int)
del df_f

"""
This section uses the np.vstack() function to convert all of our data and features
into a 2D array, where each row corresponds to a company and each column corresponds
to a feature.
"""

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

"""
This section loads the predictive score for each feature, and then sorts them,
then selects the top-K. It then keeps track of what these indices are, 
so we can filter by them.
"""

saved = joblib.load(args.load_model)
scores = saved["scores"]
ranked = np.argsort(scores)[::-1] # All negative predictive scores are excluded.
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)

"""
This section selects only those features that were deemed important above,
and then weights them by their predictive score if this is chosen as a parameter.

It also computes the norm for each company, which is used with alpha.
"""

selected_features = feat_matrix[:, selected].copy()
if args.score_weight:
    weights = scores[selected].astype(np.float32)
    selected_features *= weights[np.newaxis, :]
norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10)

"""
This section maps and merges each company and year to its corresponding row.
"""

company_ids = df["__index_level_0__"].values
feat_idx_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

pairs_B = pairs_df.merge( # This process is done twice, once for each company.
    feat_idx_df.rename(columns={"__index_level_0__": "Company1", "feat_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_B = pairs_B.merge(
    feat_idx_df.rename(columns={"__index_level_0__": "Company2", "feat_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)

"""
This section, using the idx1 and idx2 indices determined above, as well as the
input alpha value (where 0 is equivalent to a dot product, and 1 is equivalent to
a cosine similarity), calculates the similarity for each company pair.

This is done with the weighted features if specified above. 
"""

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

"""
This section uses the correlation-at-k evaluation metric (which was used for
the parent paper in an earlier section) for our new approach. 
"""

test_mask_B = pairs_B["year"].isin(test_years).values # This section isolates only the test years 
test_sims_B = cos_sims_B[test_mask_B]                 # for evaluation, using the 75-25 split from before.
test_corrs_B = corrs_B[test_mask_B]
test_sorted_B = np.argsort(test_sims_B)[::-1] 

rho_B, pval_B = spearmanr(test_sims_B, test_corrs_B)

prec_B = {}
for pct in PCTS:                                             # See the above section for more details on
    n_top = max(1, int(len(test_sims_B) * pct / 100.0))      # how the correlation-at-k metric is derived.
    prec_B[pct] = test_corrs_B[test_sorted_B[:n_top]].mean()

del feat_matrix, selected_features
gc.collect()

"""
This section computes the SIC baseline equivalent of the correlation-at-k
evaluation metric. Since the SIC codes do not produce a continuous similarity
value, we instead take only the companies for which the codes match, and 
calculate their actual similarities (which is treated as equivalent to the
top-1% from above).
"""

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
pct_same_test = 100.0 * n_same_test / n_total_test_sic # This section determines how many of the possible
                                                       # company pairs have the same SIC codes. 
sic_corr_test = ds_sic_test.loc[same_sic_test, "correlation"].mean() # This computes what is stated above: the average
                                                                     # mean correlation for only these matching pairs.

"""
This section until the end simply prints out the tables that show the results. 
"""

BOLD = "\033[1m"
RESET = "\033[0m"
yr_min, yr_max = min(test_years), max(test_years)

spearman_entries = [
    ("New approach (k={}, \u03b1={})".format(n_selected, args.norm_alpha), rho_B, pval_B),
    ("Parent paper (PCA 4000-dim)", rho_A, pval_A),
]
best_rho = max(e[1] for e in spearman_entries)

spearman_table = []
for name, rho, pval in spearman_entries:
    rho_s = f"{rho:.4f}"
    if abs(rho - best_rho) < 1e-8:
        rho_s = f"{BOLD}{rho_s}{RESET}"
    spearman_table.append([name, rho_s, f"{pval:.2e}"])

prec_headers = ["Cutoff", "New approach", "Parent paper"]
prec_headers.append("SIC baseline")

def _b(v, best):
    s = f"{v:.4f}"
    return f"{BOLD}{s}{RESET}" if abs(v - best) < 1e-8 else s

prec_table = []
for pct in PCTS:
    sic_val = sic_corr_test if pct == 1.0 else None
    row_vals = [prec_B[pct], prec_A[pct]]
    if sic_val is not None:
        row_vals.append(sic_val)
    best_val = max(row_vals)

    row = [f"top {pct:.1f}%", _b(prec_B[pct], best_val), _b(prec_A[pct], best_val)]
    row.append(_b(sic_val, best_val) if sic_val is not None else "")
    prec_table.append(row)

print()
print("Spearman rho (OOS)")
print(tabulate(
    spearman_table,
    headers=["Approach", "Spearman rho", "p-value"],
    tablefmt="simple_outline",
))

print()
print(f"Correlation-at-k, OOS {yr_min}-{yr_max}")
print(tabulate(
    prec_table,
    headers=prec_headers,
    tablefmt="simple_outline",
))

print()
print(f"Population mean correlation: {pop_corr:.4f}")
print()
print(f"New approach: k={n_selected}, "
      f"score-weighted={'yes' if args.score_weight else 'no'}, "
      f"\u03b1={args.norm_alpha}; "
      f"{len(test_sims_B):,d} OOS pairs")
print(f"Original approach: PCA 4000-dim cosine; "
      f"{len(test_sims_A):,d} OOS pairs")
