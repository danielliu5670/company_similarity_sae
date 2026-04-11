#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict
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
P.add_argument("--norm-alpha", type=float, nargs="+", default=[1.0],
               help="Norm exponent(s) for similarity: 0=dot product, 1=cosine (space-separated)")
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

df_f = pd.read_pickle(args.features_pkl)


def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


df_f["features"] = df_f["features"].apply(unwrap_feature)

df_c = load_dataset(args.cov_ds)["train"].to_pandas()

df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

if len(feat_matrix) == 0:
    raise RuntimeError("No valid feature rows remain after NaN removal.")

feat_dim = feat_matrix.shape[1]

pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)

all_years = sorted(pairs_df["year"].unique())
n_total = len(all_years)
split_idx = int(0.75 * n_total)
train_years = set(all_years[:split_idx])
test_years = set(all_years[split_idx:])

company_ids = df["__index_level_0__"].values
company_to_idx = {c: i for i, c in enumerate(company_ids)}

train_pairs = pairs_df[pairs_df["year"].isin(train_years)].copy()

valid_mask = (
    train_pairs["Company1"].isin(company_to_idx)
    & train_pairs["Company2"].isin(company_to_idx)
)
train_pairs = train_pairs[valid_mask].reset_index(drop=True)

idx1_train = train_pairs["Company1"].map(company_to_idx).values.astype(np.int64)
idx2_train = train_pairs["Company2"].map(company_to_idx).values.astype(np.int64)
corr_train = train_pairs["correlation"].values.astype(np.float32)
pop_mean = corr_train.mean()

if args.load_model is not None:
    saved = joblib.load(args.load_model)
    scores = saved["scores"]
    support = saved["support"]
else:
    binary_matrix = (feat_matrix > 0)

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

feat_df = pd.DataFrame({
    "__index_level_0__": company_ids,
    "year": df["year"].values,
    "feat_idx": np.arange(len(df)),
})

pairs_merged = pairs_df.merge(
    feat_df.rename(columns={"__index_level_0__": "Company1", "feat_idx": "idx1"}),
    on=["Company1", "year"], how="inner",
)
pairs_merged = pairs_merged.merge(
    feat_df.rename(columns={"__index_level_0__": "Company2", "feat_idx": "idx2"}),
    on=["Company2", "year"], how="inner",
)

from scipy.stats import spearmanr

idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
correlations = pairs_merged["correlation"].values

ranked = np.argsort(scores)[::-1]
summary_rows = []
pct_thresholds = [0.5, 1.0, 2.0, 5.0, 10.0]

sic_values = df["sic_code"].values
sic_strs = np.array([str(int(float(s))).zfill(4) for s in sic_values])
sic_chars = np.array([[c for c in s] for s in sic_strs])
s1c = sic_chars[idx1]
s2c = sic_chars[idx2]
sic_sims = np.cumprod(s1c == s2c, axis=1).sum(axis=1).astype(np.float32)
sic_rho_val, _ = spearmanr(sic_sims, correlations)

test_mask_all = pairs_merged["year"].isin(test_years).values
sic_test_sims = sic_sims[test_mask_all]
test_corrs_all = correlations[test_mask_all]
sic_sorted = np.argsort(sic_test_sims)[::-1]
sic_n_top = max(1, int(len(sic_test_sims) * 1.0 / 100.0))
sic_oos_top1 = test_corrs_all[sic_sorted[:sic_n_top]].mean()

BOLD = "\033[1m"
RESET = "\033[0m"


def render_detail_table(top_k, table_rows, pct_thresholds):
    best_all = {p: max(r["all_prec"][p] for r in table_rows) for p in pct_thresholds}
    best_oos = {p: max(r["oos_prec"][p] for r in table_rows) for p in pct_thresholds}

    W = [7, 14, 22, 22]

    def hline(l, m, r):
        return f"{l}{'─' * W[0]}{m}{'─' * W[1]}{m}{'─' * W[2]}{m}{'─' * W[3]}{r}"

    print(f"\ntop-k = {top_k}\n")
    print(hline("┌", "┬", "┐"))
    print(
        f"│{' alpha':<{W[0]}}│{' Spearman rho':<{W[1]}}"
        f"│{' Precision-at-k (all)':<{W[2]}}│{' Precision-at-k (OOS)':<{W[3]}}│"
    )

    for row in table_rows:
        print(hline("├", "┼", "┤"))
        for i, pct in enumerate(pct_thresholds):
            if i == 0:
                a_s = f" {str(row['alpha']):<{W[0] - 1}}"
                rho_fmt = f"{row['rho']:.4f}"
                r_s = f" {rho_fmt:<{W[1] - 1}}"
            else:
                a_s = " " * W[0]
                r_s = " " * W[1]

            all_v = row["all_prec"][pct]
            oos_v = row["oos_prec"][pct]
            all_t = f"Top {pct:5.1f}% = {all_v:.4f}"
            oos_t = f"Top {pct:5.1f}% = {oos_v:.4f}"

            all_c = f" {all_t:<{W[2] - 1}}"
            oos_c = f" {oos_t:<{W[3] - 1}}"

            if all_v == best_all[pct]:
                all_c = f" {BOLD}{all_t}{RESET}{' ' * (W[2] - 1 - len(all_t))}"
            if oos_v == best_oos[pct]:
                oos_c = f" {BOLD}{oos_t}{RESET}{' ' * (W[3] - 1 - len(oos_t))}"

            print(f"│{a_s}│{r_s}│{all_c}│{oos_c}│")

    print(hline("└", "┴", "┘"))


for top_k in args.top_k:
    selected = ranked[:top_k]
    selected = selected[scores[selected] > 0]
    selected = np.sort(selected)

    if len(selected) == 0:
        continue

    selected_features = feat_matrix[:, selected].copy()
    if args.score_weight:
        weights = scores[selected].astype(np.float32)
        selected_features *= weights[np.newaxis, :]

    raw_norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10)

    table_rows = []

    for alpha in args.norm_alpha:
        if alpha == 0:
            norm_factors = np.ones(len(selected_features), dtype=np.float32)
        else:
            norm_factors = raw_norms ** alpha

        cos_sims = np.empty(len(pairs_merged), dtype=np.float32)
        batch = 500_000
        for s in range(0, len(pairs_merged), batch):
            e = min(s + batch, len(pairs_merged))
            i1, i2 = idx1[s:e], idx2[s:e]
            cos_sims[s:e] = (
                (selected_features[i1] * selected_features[i2]).sum(1)
                / (norm_factors[i1] * norm_factors[i2])
            )

        rho, pval = spearmanr(cos_sims, correlations)

        sorted_indices = np.argsort(cos_sims)[::-1]
        all_prec = {}
        for pct in pct_thresholds:
            n_top = max(1, int(len(cos_sims) * pct / 100.0))
            all_prec[pct] = correlations[sorted_indices[:n_top]].mean()

        test_mask = pairs_merged["year"].isin(test_years).values
        test_sims = cos_sims[test_mask]
        test_corrs = correlations[test_mask]
        test_sorted = np.argsort(test_sims)[::-1]

        oos_prec = {}
        oos_top1_corr = np.nan
        for pct in pct_thresholds:
            n_top = max(1, int(len(test_sims) * pct / 100.0))
            oos_prec[pct] = test_corrs[test_sorted[:n_top]].mean()
            if pct == 1.0:
                oos_top1_corr = oos_prec[pct]

        table_rows.append({
            "alpha": alpha, "rho": rho,
            "all_prec": all_prec, "oos_prec": oos_prec,
        })

        summary_rows.append({
            "top_k": top_k, "alpha": alpha, "n_selected": len(selected),
            "spearman_rho": rho, "oos_top1pct_corr": oos_top1_corr,
        })

        out_pairs_df = pairs_merged.drop(columns=["idx1", "idx2"]).copy()
        out_pairs_df["cosine_similarity"] = cos_sims

        if len(args.top_k) == 1 and len(args.norm_alpha) == 1:
            out_path = args.out_pairs
        else:
            base, ext = os.path.splitext(args.out_pairs)
            out_path = f"{base}_k{top_k}_a{alpha}{ext}"

        out_pairs_df.to_pickle(out_path)

    render_detail_table(top_k, table_rows, pct_thresholds)

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

if summary_rows:
    best_by_k = defaultdict(lambda: -np.inf)
    for r in summary_rows:
        if r["oos_top1pct_corr"] > best_by_k[r["top_k"]]:
            best_by_k[r["top_k"]] = r["oos_top1pct_corr"]

    SW = [7, 7, 8, 16]

    def shline(l, m, r):
        return f"{l}{'─' * SW[0]}{m}{'─' * SW[1]}{m}{'─' * SW[2]}{m}{'─' * SW[3]}{r}"

    print(f"\nSummary:\n")
    print(shline("┌", "┬", "┐"))
    print(
        f"│{' top k':<{SW[0]}}│{' alpha':<{SW[1]}}"
        f"│{' rho':<{SW[2]}}│{' top 1.0% (OOS)':<{SW[3]}}│"
    )
    print(shline("├", "┼", "┤"))

    prev_k = None
    for r in summary_rows:
        if prev_k is not None and r["top_k"] != prev_k:
            print(shline("├", "┼", "┤"))

        k_fmt = str(r["top_k"])
        a_fmt = f"{r['alpha']:.2f}"
        rho_fmt = f"{r['spearman_rho']:.4f}"
        oos_fmt = f"{r['oos_top1pct_corr']:.4f}"

        k_s = f" {k_fmt:<{SW[0] - 1}}"
        a_s = f" {a_fmt:<{SW[1] - 1}}"
        rho_s = f" {rho_fmt:<{SW[2] - 1}}"

        if r["oos_top1pct_corr"] == best_by_k[r["top_k"]]:
            oos_s = f" {BOLD}{oos_fmt}{RESET}{' ' * (SW[3] - 1 - len(oos_fmt))}"
        else:
            oos_s = f" {oos_fmt:<{SW[3] - 1}}"

        print(f"│{k_s}│{a_s}│{rho_s}│{oos_s}│")
        prev_k = r["top_k"]

    print(shline("├", "┼", "┤"))
    sic_rho_fmt = f"{sic_rho_val:.4f}"
    sic_oos_fmt = f"{sic_oos_top1:.4f}"
    print(
        f"│{' SIC':<{SW[0]}}│{' ' * SW[1]}"
        f"│ {sic_rho_fmt:<{SW[2] - 1}}│ {sic_oos_fmt:<{SW[3] - 1}}│"
    )
    print(shline("└", "┴", "┘"))
