"""
This file generates the hexbin plot that is referenced in our paper.
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datasets import load_dataset
from scipy.stats import spearmanr
import gc
import os


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
P.add_argument("--pair-batch", type=int, default=500_000)
P.add_argument("--out-dir", default=".")
P.add_argument("--scatter", action="store_true",
               help="Also produce a scatter plot with low alpha")
P.add_argument("--scatter-alpha", type=float, default=0.002,
               help="Transparency for scatter points (default: 0.002)")
P.add_argument("--scatter-sample", type=int, default=2_000_000,
               help="Max points to plot in scatter (default: 2M)")
P.add_argument("--oos-only", action="store_true",
               help="Restrict to OOS years only")
P.add_argument("--dpi", type=int, default=200)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--original-pairs-ds", default="v1ctor10/cos_sim_4000pca_exp")
args = P.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

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

feat_norms = np.linalg.norm(selected_features, axis=1).clip(min=1e-10).astype(np.float32)

del feat_matrix; gc.collect()

pairs_df = load_dataset(args.original_pairs_ds)["train"].to_pandas()
pairs_df = pairs_df.dropna(subset=["correlation"])
pairs_df["year"] = pairs_df["year"].astype(int)
pairs_df["Company1"] = pairs_df["Company1"].astype(str)
pairs_df["Company2"] = pairs_df["Company2"].astype(str)

all_years = sorted(pairs_df["year"].unique())
split_idx = int(0.75 * len(all_years))
test_years = set(all_years[split_idx:])

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

if args.oos_only:
    pairs_merged = pairs_merged[pairs_merged["year"].isin(test_years)].reset_index(drop=True)

idx1 = pairs_merged["idx1"].values
idx2 = pairs_merged["idx2"].values
n_pairs = len(pairs_merged)

sims = np.empty(n_pairs, dtype=np.float32)
batch = args.pair_batch
for s in range(0, n_pairs, batch):
    e = min(s + batch, n_pairs)
    i1, i2 = idx1[s:e], idx2[s:e]
    sims[s:e] = (selected_features[i1] * selected_features[i2]).sum(axis=1)

norm_products = feat_norms[idx1] * feat_norms[idx2]

r_squared = np.corrcoef(norm_products, sims)[0, 1] ** 2
rho, _ = spearmanr(norm_products, sims)
slope = np.cov(norm_products, sims)[0, 1] / np.var(norm_products)
intercept = sims.mean() - slope * norm_products.mean()

subset_label = "OOS" if args.oos_only else "all years"
print(f"\n  R²: {r_squared:.4f}")
print(f"  Pearson r: {np.sqrt(r_squared):.4f}")
print(f"  Spearman rho: {rho:.4f}")
print(f"  Regression: y = {slope:.6f} * x + {intercept:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))

hb = ax.hexbin(
    norm_products, sims,
    gridsize=150,
    cmap="viridis",
    mincnt=1,
    norm=mcolors.LogNorm(),
    linewidths=0.1,
)

x_range = np.linspace(norm_products.min(), norm_products.max(), 200)
y_pred = slope * x_range + intercept
ax.plot(x_range, y_pred, color="red", linewidth=1.5, linestyle="--",
        label=f"OLS fit (R² = {r_squared:.3f})")

ax.set_xlabel("Norm", fontsize=12)
ax.set_ylabel("Similarity", fontsize=12)
ax.set_title(f"Norm vs. Similarity", fontsize=13)
ax.legend(fontsize=10, loc="upper left")

cb = fig.colorbar(hb, ax=ax)
cb.set_label("Log count", fontsize=10)

fig.tight_layout()
out_hex = os.path.join(args.out_dir, "norm_vs_similarity_hexbin.png")
fig.savefig(out_hex, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)

if args.scatter:
    n_plot = min(args.scatter_sample, n_pairs)
    if n_plot < n_pairs:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_pairs, size=n_plot, replace=False)
        x_plot = norm_products[sample_idx]
        y_plot = sims[sample_idx]
        sample_note = f" (random {n_plot:,d} of {n_pairs:,d})"
    else:
        x_plot = norm_products
        y_plot = sims
        sample_note = ""

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x_plot, y_plot,
               s=0.5,
               alpha=args.scatter_alpha,
               color="steelblue",
               rasterized=True)

    ax.plot(x_range, y_pred, color="red", linewidth=1.5, linestyle="--",
            label=f"OLS trend line, R^2 = {r_squared:.3f})")

    ax.set_xlabel("Norm product  (||a|| · ||b||)", fontsize=12)
    ax.set_ylabel("Dot product similarity", fontsize=12)
    ax.set_title(f"Norm product vs. dot product similarity{sample_note}",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    out_scat = os.path.join(args.out_dir, "norm_vs_similarity_scatter.png")
    fig.savefig(out_scat, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)