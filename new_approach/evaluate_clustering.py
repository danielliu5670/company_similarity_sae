#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
from datasets import load_dataset
from sklearn.cluster import SpectralClustering
from tabulate import tabulate
from tqdm import tqdm
import gc

warnings.filterwarnings("ignore", category=UserWarning)

"""
This function essentially "peels back" all of the potential nesting locations 
that the features could be stored in. It then takes these features and 
returns them as a 1D array.
"""

def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


"""
This function builds a dense N x N affinity matrix for a single year, using
the specified similarity column from the pairs dataframe. Negative similarities
are clipped to zero, since spectral clustering requires non-negative affinities.

It returns the affinity matrix and the ordered list of company identifiers
corresponding to its rows and columns.
"""

def build_affinity_matrix(pairs_df, year, sim_col):
    ydf = pairs_df[pairs_df["year"] == year]
    if ydf.empty:
        return None, None

    companies = sorted(
        set(ydf["Company1"].values).union(ydf["Company2"].values)
    )
    n = len(companies)
    comp_to_idx = {c: i for i, c in enumerate(companies)}

    A = np.zeros((n, n), dtype=np.float32)
    c1_arr = ydf["Company1"].values
    c2_arr = ydf["Company2"].values
    sims = np.maximum(
        ydf[sim_col].values.astype(np.float32), 0.0
    )

    for a, b, s in zip(c1_arr, c2_arr, sims):
        i = comp_to_idx[a]
        j = comp_to_idx[b]
        A[i, j] = s
        A[j, i] = s

    np.fill_diagonal(A, A.diagonal() + 1e-10) # For numerical stability.
    return A, companies


"""
This function runs spectral clustering on a single year's affinity matrix.
The output is a dictionary where each key is a cluster identifier and each 
value is the sorted list of companies in that cluster.
"""

def spectral_cluster_year(affinity, companies, n_clusters, random_state=42):
    n = len(companies)
    if n <= n_clusters or n < 2:
        return None

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
        n_init=10,
    )
    try:
        labels = sc.fit_predict(affinity)
    except Exception:
        return None

    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab) + 1, []).append(companies[i])
    for key in clusters:
        clusters[key] = sorted(clusters[key])
    return clusters


"""
This function applies spectral clustering across a collection of years, using 
the same k in each year. It returns a dictionary mapping each year to its 
per-year clusters dictionary (or None if clustering failed for that year).
"""

def run_spectral_all_years(pairs_df, years, sim_col, n_clusters, desc=None):
    results = {}
    iterator = tqdm(years, desc=desc) if desc else years
    for year in iterator:
        A, companies = build_affinity_matrix(pairs_df, year, sim_col)
        if A is None:
            results[year] = None
            continue
        results[year] = spectral_cluster_year(A, companies, n_clusters)
    return results


"""
This function creates a dictionary that can be used for a speedy lookup, where,
provided any two companies, you can instantly get back their return correlation.
"""

def build_pair_lookup(pairs_df, years):
    lookup = {}
    for year in years:
        ydf = pairs_df[pairs_df["year"] == year]
        c1s = ydf["Company1"].values
        c2s = ydf["Company2"].values
        corrs = ydf["correlation"].values
        d = {}
        for i in range(len(c1s)):
            d[frozenset((c1s[i], c2s[i]))] = float(corrs[i])
        lookup[year] = d
    return lookup


"""
This function evaluates the mean correlation of the clusters for a single year.
It returns the unweighted and size-weighted mean correlations, alongside the
cluster count, total intra-cluster pair count, and the list of cluster sizes.
"""

def evaluate_year_clusters(ylookup, clusters):
    cluster_means = []
    cluster_sizes = []
    total_corr = 0.0
    total_pairs = 0

    for companies in clusters.values():
        if len(companies) <= 1:
            continue
        corrs = []
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                c = ylookup.get(frozenset((companies[i], companies[j])))
                if c is not None:
                    corrs.append(c)
        if corrs:
            cluster_means.append(np.mean(corrs))
            cluster_sizes.append(len(companies))
            total_corr += sum(corrs)
            total_pairs += len(corrs)

    mc_uw = np.mean(cluster_means) if cluster_means else np.nan
    mc_pw = (total_corr / total_pairs) if total_pairs > 0 else np.nan
    return mc_uw, mc_pw, len(cluster_means), total_pairs, cluster_sizes


"""
This function takes the per-year information from above and returns the 
overall measures across years, along with a list of per-year detail dicts.
"""

def evaluate_clusters(pair_lookup, year_clusters):
    uw_vals, pw_vals = [], []
    details = []

    for year, clusters in year_clusters:
        if clusters is None:
            continue
        ylookup = pair_lookup.get(year, {})
        mc_uw, mc_pw, n_cl, n_pairs, sizes = evaluate_year_clusters(
            ylookup, clusters
        )
        uw_vals.append(mc_uw)
        pw_vals.append(mc_pw)
        details.append({
            "year": year,
            "n_clusters": n_cl,
            "n_intra_pairs": n_pairs,
            "mc_unweighted": mc_uw,
            "mc_pair_weighted": mc_pw,
            "median_size": int(np.median(sizes)) if sizes else 0,
            "max_size": max(sizes) if sizes else 0,
            "n_covered": sum(sizes),
        })

    valid_uw = [x for x in uw_vals if not np.isnan(x)]
    valid_pw = [x for x in pw_vals if not np.isnan(x)]
    return (
        np.mean(valid_uw) if valid_uw else np.nan,
        np.mean(valid_pw) if valid_pw else np.nan,
        details,
    )


"""
This function, provided a set of k values, iterates through each one and 
determines which produces the highest unweighted mean return correlation 
on the training years. It then returns this best k value, as well as all
sweep results.
"""

def sweep_k(pairs_df, pair_lookup, years, sim_col, k_values, label):
    best_k = None
    best_mc = -np.inf
    results = []

    for k in k_values:
        year_clusters = run_spectral_all_years(
            pairs_df, years, sim_col, k, desc=f"{label} k={k}",
        )
        yc_list = [(y, c) for y, c in year_clusters.items() if c is not None]
        mc_uw, mc_pw, _ = evaluate_clusters(pair_lookup, yc_list)
        results.append({
            "k": k,
            "mc_unweighted": mc_uw,
            "mc_pair_weighted": mc_pw,
        })
        print(f"    k={k}: unweighted MC={mc_uw:.4f}, weighted MC={mc_pw:.4f}")
        if not np.isnan(mc_uw) and mc_uw > best_mc:
            best_mc = mc_uw
            best_k = k

    return best_k, results


"""
These are all the command-line arguments that can be passed into the file.
"""

P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--top-k", type=int, default=1000)
P.add_argument("--score-weight", action="store_true", default=True)
P.add_argument("--no-score-weight", action="store_false", dest="score_weight")
P.add_argument("--norm-alpha", type=float, default=0.0)
P.add_argument(
    "--k-values", type=int, nargs="+",
    default=[5, 10, 20, 40, 60, 80, 100, 150, 200],
    help="Space-separated list of cluster counts to sweep."
)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--original-pairs-ds", default="v1ctor10/cos_sim_4000pca_exp")
args = P.parse_args()


"""
This section pulls the company pairs from HuggingFace, drops any NA values, and
standardizes the columns to all be of the same type (int, str, str).
"""

raw = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = raw.dropna(subset=["correlation", "cosine_similarity"]).copy()
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)
del raw; gc.collect()


"""
This section creates the training / testing split, specifically being 75/25 across
years. This is not done across rows to avoid data leakage and various other problems.

It also defines the population mean correlation (pop_corr).
"""

all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
train_years = all_years[:split_idx]
test_years = all_years[split_idx:]
pop_corr = pairs_df["correlation"].mean()


"""
This section pulls both the company metadata and the feature vectors from the 
input pickle file (see collect_features.py). It then, for each company, merges
both of these sources.
"""

df_c = load_dataset(args.cov_ds)["train"].to_pandas()
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)
del df_f; gc.collect()


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
ranked = np.argsort(scores)[::-1]
selected = ranked[:args.top_k]
selected = selected[scores[selected] > 0]
selected = np.sort(selected)
n_selected = len(selected)


"""
This section selects only those features that were deemed important above,
and then weights them by their predictive score if this is chosen as a parameter.

It also computes the norm for each company, which is used with alpha.
"""

selected_features = feat_matrix[:, selected].copy().astype(np.float32)
if args.score_weight:
    weights = scores[selected].astype(np.float32)
    selected_features *= weights[np.newaxis, :]
norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10).astype(np.float32)
del feat_matrix; gc.collect()


"""
This section maps and merges each company and year to its corresponding row.
"""

company_ids = df["__index_level_0__"].values
feat_idx_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

pairs_merged = pairs_df.merge(
    feat_idx_df.rename(columns={
        "__index_level_0__": "Company1", "feat_idx": "idx1",
    }),
    on=["Company1", "year"], how="inner",
)
pairs_merged = pairs_merged.merge(
    feat_idx_df.rename(columns={
        "__index_level_0__": "Company2", "feat_idx": "idx2",
    }),
    on=["Company2", "year"], how="inner",
)


"""
This section, using the idx1 and idx2 indices determined above, as well as the
input alpha value (where 0 is equivalent to a dot product, and 1 is equivalent to
a cosine similarity), calculates the similarity for each company pair.

This is done with the weighted features if specified above.
"""

idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
alpha = args.norm_alpha

sims_ours = np.empty(len(pairs_merged), dtype=np.float32)
batch = 500_000
for s in range(0, len(pairs_merged), batch):
    e = min(s + batch, len(pairs_merged))
    i1, i2 = idx1[s:e], idx2[s:e]
    dot = (selected_features[i1] * selected_features[i2]).sum(1)
    if alpha > 0:
        sims_ours[s:e] = dot / ((norms[i1] * norms[i2]) ** alpha)
    else:
        sims_ours[s:e] = dot

pairs_merged["sim_ours"] = sims_ours
del selected_features; gc.collect()


"""
This section constructs the pair lookup dictionary across all years for fast
correlation retrieval during cluster evaluation.
"""

pair_lookup = build_pair_lookup(pairs_merged, all_years)


"""
This section runs spectral clustering for the parent paper's PCA cosine similarity.
It sweeps k on the training years, then applies the best k to the test years and
all years for evaluation.
"""

print("\nParent paper (PCA cosine) - sweeping k on train years:")
best_k_A, sweep_A = sweep_k(
    pairs_merged, pair_lookup, train_years,
    "cosine_similarity", args.k_values, label="Parent",
)
print(f"  Best k for parent paper: {best_k_A}")

test_clusters_A = run_spectral_all_years(
    pairs_merged, test_years, "cosine_similarity", best_k_A,
    desc="Parent test",
)
all_clusters_A = run_spectral_all_years(
    pairs_merged, all_years, "cosine_similarity", best_k_A,
    desc="Parent all",
)
test_cl_A = [(y, c) for y, c in test_clusters_A.items() if c is not None]
all_cl_A = [(y, c) for y, c in all_clusters_A.items() if c is not None]

mc_uw_A_oos, mc_pw_A_oos, det_A_oos = evaluate_clusters(pair_lookup, test_cl_A)
mc_uw_A_all, mc_pw_A_all, _ = evaluate_clusters(pair_lookup, all_cl_A)


"""
This section does the same thing but for our new approach's supervised SAE 
dot product similarity.
"""

print("\nNew method (supervised SAE) - sweeping k on train years:")
best_k_B, sweep_B = sweep_k(
    pairs_merged, pair_lookup, train_years,
    "sim_ours", args.k_values, label="New",
)
print(f"  Best k for new method: {best_k_B}")

test_clusters_B = run_spectral_all_years(
    pairs_merged, test_years, "sim_ours", best_k_B,
    desc="New test",
)
all_clusters_B = run_spectral_all_years(
    pairs_merged, all_years, "sim_ours", best_k_B,
    desc="New all",
)
test_cl_B = [(y, c) for y, c in test_clusters_B.items() if c is not None]
all_cl_B = [(y, c) for y, c in all_clusters_B.items() if c is not None]

mc_uw_B_oos, mc_pw_B_oos, det_B_oos = evaluate_clusters(pair_lookup, test_cl_B)
mc_uw_B_all, mc_pw_B_all, _ = evaluate_clusters(pair_lookup, all_cl_B)


"""
This section determines the mean correlation and other metrics for the SIC baseline.
Since it doesn't have the same structure as the other methods, we simply construct
clusters where each one matches in SIC code, but use the same metrics.
"""

sic_info = df_c[["__index_level_0__", "year", "sic_code"]].dropna(
    subset=["sic_code"]
).copy()
sic_info["year"] = sic_info["year"].astype(int)
sic_dedup = sic_info.drop_duplicates(
    subset=["__index_level_0__", "year"], keep="last",
)

sic_year_clusters = []
for year in all_years:
    yd = sic_dedup[sic_dedup["year"] == year]
    groups = yd.groupby("sic_code")["__index_level_0__"].apply(
        sorted
    ).to_dict()
    clusters = {i: v for i, (_, v) in enumerate(groups.items(), 1)}
    sic_year_clusters.append((year, clusters))

sic_test = [(y, c) for y, c in sic_year_clusters if y in test_years]
mc_uw_C_oos, mc_pw_C_oos, det_C_oos = evaluate_clusters(pair_lookup, sic_test)
mc_uw_C_all, mc_pw_C_all, _ = evaluate_clusters(pair_lookup, sic_year_clusters)


"""
This section until the end simply prints out the tables that show the results.
"""

def _bold_best(rows, oos_col_idx=1):
    best_val = max(float(r[oos_col_idx]) for r in rows)
    out = []
    for r in rows:
        if float(r[oos_col_idx]) == best_val:
            out.append([f"\033[1m{c}\033[0m" for c in r])
        else:
            out.append(list(r))
    return out


print(f"\nUnweighted MC(Gk) - Spectral Clustering")
uw_table = [
    [f"New method (k={best_k_B})",
     f"{mc_uw_B_oos:.4f}", f"{mc_uw_B_all:.4f}"],
    [f"Parent paper (k={best_k_A})",
     f"{mc_uw_A_oos:.4f}", f"{mc_uw_A_all:.4f}"],
    ["SIC code",
     f"{mc_uw_C_oos:.4f}", f"{mc_uw_C_all:.4f}"],
    ["Population mean",
     f"{pop_corr:.4f}", f"{pop_corr:.4f}"],
]
print(tabulate(
    _bold_best(uw_table),
    headers=["Approach", "OOS", "All years"],
    tablefmt="simple_outline",
))

print(f"\nWeighted MC(Gk) - Spectral Clustering")
pw_table = [
    [f"New method (k={best_k_B})",
     f"{mc_pw_B_oos:.4f}", f"{mc_pw_B_all:.4f}"],
    [f"Parent paper (k={best_k_A})",
     f"{mc_pw_A_oos:.4f}", f"{mc_pw_A_all:.4f}"],
    ["SIC code",
     f"{mc_pw_C_oos:.4f}", f"{mc_pw_C_all:.4f}"],
    ["Population mean",
     f"{pop_corr:.4f}", f"{pop_corr:.4f}"],
]
print(tabulate(
    _bold_best(pw_table),
    headers=["Approach", "OOS", "All years"],
    tablefmt="simple_outline",
))


def print_yearly(label, details):
    print(f"\n{label}")
    rows = []
    for d in sorted(details, key=lambda x: x["year"]):
        rows.append([
            d["year"],
            d["n_clusters"],
            f"{d['n_intra_pairs']:,d}",
            f"{d['mc_unweighted']:.4f}",
            f"{d['mc_pair_weighted']:.4f}",
            d["median_size"],
            d["max_size"],
            d["n_covered"],
        ])
    print(tabulate(
        rows,
        headers=["Year", "Clust", "Pairs", "MC unwtd", "MC wtd",
                 "Med sz", "Max sz", "Covered"],
        tablefmt="simple_outline",
    ))


print_yearly(f"New method OOS (spectral, k={best_k_B}):", det_B_oos)
print_yearly(f"Parent paper OOS (spectral, k={best_k_A}):", det_A_oos)
print_yearly("SIC baseline OOS:", det_C_oos)

print(f"\nSweep summary (train years)")
sweep_rows = []
by_k_A = {r["k"]: r["mc_unweighted"] for r in sweep_A}
by_k_B = {r["k"]: r["mc_unweighted"] for r in sweep_B}
for k in args.k_values:
    sweep_rows.append([
        k,
        f"{by_k_A.get(k, np.nan):.4f}",
        f"{by_k_B.get(k, np.nan):.4f}",
    ])
print(tabulate(
    sweep_rows,
    headers=["k", "Parent paper (uw MC)", "New method (uw MC)"],
    tablefmt="simple_outline",
))