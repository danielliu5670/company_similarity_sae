#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import joblib
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tqdm import tqdm
import gc
import time

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
This function uses the networkx library that was imported above, and constructs
a Minimum Spanning Tree from a graph. Essentially, each edge in a graph connects 
two companies, and its weight is their distance. 

Afterward, it extracts the MST using some built-in functions from networkx.
"""

def build_msts(pairs_df, years, distance_col):
    msts = {}
    for year in tqdm(years, desc="Building MSTs"):
        ydf = pairs_df[pairs_df["year"] == year]
        if ydf.empty:
            msts[year] = None
            continue
        G = nx.Graph()
        G.add_weighted_edges_from(zip(
            ydf["Company1"].values,
            ydf["Company2"].values,
            ydf[distance_col].values.astype(float),
        ))
        if G.number_of_edges() == 0:
            msts[year] = None
            continue
        msts[year] = nx.minimum_spanning_tree(G, weight="weight") # Notice that the output is actually a
                                                                  # dictionary, where each of the keys
                                                                  # is a year, and the value is the MST.
    return msts

"""
This function, for a provided theta, cuts all of the edges that have a theta 
above that threshold. Then, each of the clusters that are formed from this
are given an "ID" and then returned.
"""

def clusters_at_theta(mst, theta):
    if mst is None:
        return {}
    g = mst.copy() # The graph is copied so the original MST is not disturbed, in case it is needed.
    g.remove_edges_from([
        (u, v) for u, v, d in g.edges(data=True) if d["weight"] > theta
    ])
    return {                                                 # This is the dictionary that is returned,
        i: sorted(list(c))                                   # where each of the keys is the "ID" of the 
        for i, c in enumerate(nx.connected_components(g), 1) # cluster and the value is the cluster itself.
    }


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
            d[frozenset((c1s[i], c2s[i]))] = float(corrs[i]) # This function uses a frozenset to do this!
        lookup[year] = d
    return lookup

"""
This function evaluates the mean correlation of the clusters formed above,
per year.  Specifically, it iterates through all of the clusters in a year, 
and then  extracts all of the correlations, pairs, and other information.

Afterward, it calculates both the unweighted mean correlation and
weighted correlation (by the size of the cluster). It also
returns the number of cluseters, pairs, and the cluster sizes.
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

    mc_uw = np.mean(cluster_means) if cluster_means else np.nan       # This calculates the unweighted MC, whereas...
    mc_pw = (total_corr / total_pairs) if total_pairs > 0 else np.nan # ... this calculates the weighted MC.
    return mc_uw, mc_pw, len(cluster_means), total_pairs, cluster_sizes 

"""
This function takes the per-year information calculated above and returns
the overall measures. 

This includes the average unweighted correlation, average weighted correlation, 
number of pairs, number of clusters, median and max size of clusters, and how 
many companies were covered.
"""

def evaluate_clusters(pair_lookup, year_clusters):
    uw_vals, pw_vals = [], []
    details = []

    for year, clusters in year_clusters:
        ylookup = pair_lookup.get(year, {})
        mc_uw, mc_pw, n_cl, n_pairs, sizes = evaluate_year_clusters(
            ylookup, clusters
        )
        uw_vals.append(mc_uw)
        pw_vals.append(mc_pw)
        details.append({
            "year": year,                                            # <  
            "n_clusters": n_cl,                                      #  |
            "n_intra_pairs": n_pairs,                                #  |
            "mc_unweighted": mc_uw,                                  #  | This is the total list of things 
            "mc_pair_weighted": mc_pw,                               #  | returned as details.
            "median_size": int(np.median(sizes)) if sizes else 0,    #  |
            "max_size": max(sizes) if sizes else 0,                  #  |
            "n_covered": sum(sizes),                                 # <
        })

    valid_uw = [x for x in uw_vals if not np.isnan(x)] # This step filters out all of the NAN values.
    valid_pw = [x for x in pw_vals if not np.isnan(x)]
    return (
        np.mean(valid_uw) if valid_uw else np.nan,
        np.mean(valid_pw) if valid_pw else np.nan,
        details,
    )

"""
This function, provided a set of theta values, iterates through each one and 
determines which produces the highest unweighted mean return correlation. 

It then returns this best theta value, as well as the associated unweighted
and weighted mean correlation.
"""

def sweep_theta(msts, pair_lookup, years, thetas):
    best_theta = None
    best_mc = -np.inf
    results = []

    for theta in tqdm(thetas, desc="Sweeping Theta"):
        yc = []
        for year in years:
            if msts.get(year) is not None:
                yc.append((year, clusters_at_theta(msts[year], theta)))
        mc_uw, mc_pw, _ = evaluate_clusters(pair_lookup, yc)
        results.append({
            "theta": theta,
            "mc_unweighted": mc_uw,
            "mc_pair_weighted": mc_pw, # This weighted value is not used during computation,
                                       # but it is still part of the evaluation.            
        })
        if not np.isnan(mc_uw) and mc_uw > best_mc: # This computes the maximum unweighted MC.
            best_mc = mc_uw
            best_theta = theta

    return best_theta, results 

"""
These are all the command-line arguments that can be passed into the file.
"""

P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--top-k", type=int, default=1250)
P.add_argument("--score-weight", action="store_true", default=True)
P.add_argument("--no-score-weight", action="store_false", dest="score_weight")
P.add_argument("--norm-alpha", type=float, default=0.0)
P.add_argument("--theta-min", type=float, default=-4.5)
P.add_argument("--theta-max", type=float, default=-1.0)
P.add_argument("--theta-step", type=float, default=0.1)
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
This section pulls both the company metadata and other features, as well as the 
feature vectors from the input pickle file (see collect_features.py). It then, for each
company, merges both of these sources. 
"""

df_c = load_dataset(args.cov_ds)["train"].to_pandas()
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"]) # If any of the companies does not have an SIC code,
                                    # it is dropped. This is because knowing a company's
                                    # SIC is necessary for the baseline.
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
    df = df[~nan_mask].reset_index(drop=True) # This line drops any NA values.
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

selected_features = feat_matrix[:, selected].copy().astype(np.float32)
if args.score_weight:
    weights = scores[selected].astype(np.float32)
    selected_features *= weights[np.newaxis, :] # This is the score-weighting step.
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

pairs_merged = pairs_df.merge( # This process is done twice, once for each company.
    feat_idx_df.rename(columns={
        "__index_level_0__": "Company1", "feat_idx": "idx1",
    }),
    on=["Company1", "year"], how="inner",
)
pairs_merged = pairs_merged.merge( # Company 2.
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
for s in range(0, len(pairs_merged), batch):                     # This process is done in batches so the machine 
    e = min(s + batch, len(pairs_merged))                        # does not run out of memory. Each of the company  
    i1, i2 = idx1[s:e], idx2[s:e]                                # has its features element-wise multiplied using
    dot = (selected_features[i1] * selected_features[i2]).sum(1) # fancy-indexing.
    if alpha > 0:
        sims_ours[s:e] = dot / ((norms[i1] * norms[i2]) ** alpha)
    else:
        sims_ours[s:e] = dot

pairs_merged["sim_ours"] = sims_ours
del selected_features; gc.collect()

pairs_merged["dist_pca"] = (
    1.0 - pairs_merged["cosine_similarity"].values # Additionally, since we need distances for
).astype(np.float32)                               # the MST, we convert them from similarities here.
pairs_merged["dist_ours"] = (-sims_ours).astype(np.float32)

"""
Since the theta threshold for cutting the MST needs to use data that is normalized,
this section fits a StandardScaler on the train years, and then applies it on 
the test years, so that the mean is 0 and average is 1 (for the distance weights).
"""

train_mask = pairs_merged["year"].isin(train_years)

scaler_A = StandardScaler()
scaler_A.fit(pairs_merged.loc[train_mask, ["dist_pca"]])
pairs_merged["dist_pca_s"] = scaler_A.transform(
    pairs_merged[["dist_pca"]]
).astype(np.float32).ravel()

scaler_B = StandardScaler()
scaler_B.fit(pairs_merged.loc[train_mask, ["dist_ours"]])
pairs_merged["dist_ours_s"] = scaler_B.transform(
    pairs_merged[["dist_ours"]]
).astype(np.float32).ravel()

pair_lookup = build_pair_lookup(pairs_merged, all_years)

thetas = np.arange(                                                       # This section automatically generates a set
    args.theta_min, args.theta_max + args.theta_step / 2, args.theta_step # of thetas that can be tested, according to 
)                                                                         # the user's input.
thetas = np.round(thetas, 2)

"""
This section does the entire process we just defined for our parent paper; specifically,
it builds the MSTs, determines the best thetas for the MST, and then evaluate how
this threshold performs using both the out-of-sample (test) and all years graphs.
"""

print("\nParent paper:")

msts_A = build_msts(pairs_merged, all_years, "dist_pca_s") # This builds the MSTs.
best_theta_A, sweep_A = sweep_theta(                       # This determines the best thetas.
    msts_A, pair_lookup, train_years, thetas
)

if best_theta_A is None: # Edge case.
    print("  No theta found.")
    best_theta_A = thetas[len(thetas) // 2]

test_cl_A = [ # These are the test mean correlations.
    (y, clusters_at_theta(msts_A[y], best_theta_A))
    for y in test_years if msts_A.get(y) is not None
]
mc_uw_A_oos, mc_pw_A_oos, det_A_oos = evaluate_clusters(
    pair_lookup, test_cl_A
)

all_cl_A = [ # These are the all-years mean correlations.
    (y, clusters_at_theta(msts_A[y], best_theta_A))
    for y in all_years if msts_A.get(y) is not None
]
mc_uw_A_all, mc_pw_A_all, _ = evaluate_clusters(pair_lookup, all_cl_A)
del msts_A; gc.collect()

"""
This section does the same as the previous section, except for our new approach.
"""

print(f"\nNew method:")

msts_B = build_msts(pairs_merged, all_years, "dist_ours_s")
best_theta_B, sweep_B = sweep_theta(
    msts_B, pair_lookup, train_years, thetas
)

if best_theta_B is None:
    print("  No theta found.")
    best_theta_B = thetas[len(thetas) // 2]

test_cl_B = [
    (y, clusters_at_theta(msts_B[y], best_theta_B))
    for y in test_years if msts_B.get(y) is not None
]
mc_uw_B_oos, mc_pw_B_oos, det_B_oos = evaluate_clusters(
    pair_lookup, test_cl_B
)

all_cl_B = [
    (y, clusters_at_theta(msts_B[y], best_theta_B))
    for y in all_years if msts_B.get(y) is not None
]
mc_uw_B_all, mc_pw_B_all, _ = evaluate_clusters(pair_lookup, all_cl_B)
del msts_B; gc.collect()

"""
This section determines the mean correlation and other metrics for the SCI baseline.
Since it doesn't have the same structure as the other methods, we simply construct
clusters where each one matches in SIC code, but use the same metrics.
"""

sic_info = df_c[["__index_level_0__", "year", "sic_code"]].dropna(
    subset=["sic_code"]
).copy()
sic_info["year"] = sic_info["year"].astype(int)
sic_dedup = sic_info.drop_duplicates(
    subset=["__index_level_0__", "year"], keep="last"
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
mc_uw_C_oos, mc_pw_C_oos, det_C_oos = evaluate_clusters( # Notice that the same metrics are 
    pair_lookup, sic_test                                # being evaluated.
)
mc_uw_C_all, mc_pw_C_all, _ = evaluate_clusters(
    pair_lookup, sic_year_clusters
)

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

print(f"\nUnweighted MC(Gk)")
uw_table = [
    ["New method",
     f"{mc_uw_B_oos:.4f}", f"{mc_uw_B_all:.4f}"],
    ["Parent paper",
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

print(f"\nWeighted MC(Gk)")
pw_table = [
    ["New method",
     f"{mc_pw_B_oos:.4f}", f"{mc_pw_B_all:.4f}"],
    ["Parent paper",
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


print_yearly(f"New method OOS (\u03b8={best_theta_B:.2f}):", det_B_oos)
print_yearly(f"Parent paper OOS (\u03b8={best_theta_A:.2f}):", det_A_oos)
print_yearly("SIC baseline OOS:", det_C_oos)