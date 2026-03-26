#!/usr/bin/env python3
"""
Train a sparse feature-attention model that learns which SAE/PCA dimensions
predict pairwise return correlations. Evaluate under pair-weighted metric.

Usage (Colab):
    !python train_sparse_attention.py \
        --pairs-pkl  /content/drive/MyDrive/company_similarity_sae/data/llama_pairs.pkl \
        --features-pkl /content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl \
        --pca-model  /content/drive/MyDrive/company_similarity_sae/data/llama_pca_model.pkl
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import joblib
import networkx as nx
from datasets import load_dataset
from tqdm import tqdm


# ================================================================
# Dataset
# ================================================================
class PairIndexDataset(Dataset):
    """Indexes into a shared feature matrix — avoids duplicating data."""

    def __init__(self, features, idx1, idx2, targets):
        self.features = features  # (N_companies, D) float32 numpy
        self.idx1 = idx1.astype(np.int64)
        self.idx2 = idx2.astype(np.int64)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.features[self.idx1[i]]),
            torch.from_numpy(self.features[self.idx2[i]]),
            torch.tensor(self.targets[i]),
        )


# ================================================================
# Model
# ================================================================
class SparseAttention(nn.Module):
    """
    Learns a non-negative weight per PCA dimension.
    Similarity = cosine( w * a,  w * b ).
    L1 penalty on w encourages sparsity.
    """

    def __init__(self, dim):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.ones(dim))

    def get_weights(self):
        return torch.abs(self.raw_weights)

    def forward(self, a, b):
        w = self.get_weights()
        return nn.functional.cosine_similarity(a * w, b * w, dim=-1)


# ================================================================
# Evaluation helpers
# ================================================================
def calculate_weighted_avg_correlation(pairs_df, cluster_df):
    results = []
    for _, row in cluster_df.iterrows():
        year = row["year"]
        clusters = row["clusters"]
        year_data = pairs_df[pairs_df["year"] == year]
        corr_sum = 0.0
        pair_count = 0
        for companies in clusters.values():
            if len(companies) <= 1:
                continue
            cp = year_data[
                year_data["Company1"].isin(companies)
                & year_data["Company2"].isin(companies)
            ]
            if not cp.empty:
                corr_sum += cp["correlation"].sum()
                pair_count += len(cp)
        if pair_count > 0:
            results.append({"year": year, "avg_corr": corr_sum / pair_count})
        else:
            results.append({"year": year, "avg_corr": np.nan})
    return pd.DataFrame(results)


def mst_cluster_years(pairs_df, years, threshold, distance_col):
    results = []
    for year in years:
        ydf = pairs_df[pairs_df["year"] == year]
        if ydf.empty:
            results.append({"year": year, "clusters": {}})
            continue
        G = nx.Graph()
        G.add_weighted_edges_from(
            zip(ydf["Company1"], ydf["Company2"], ydf[distance_col])
        )
        if G.number_of_edges() == 0:
            results.append({"year": year, "clusters": {}})
            continue
        mst = nx.minimum_spanning_tree(G, weight="weight")
        mst.remove_edges_from(
            [(u, v) for u, v, d in mst.edges(data=True) if d["weight"] > threshold]
        )
        clusters = {
            i: sorted(list(c))
            for i, c in enumerate(nx.connected_components(mst), 1)
        }
        results.append({"year": year, "clusters": clusters})
    return pd.DataFrame(results)


# ================================================================
# Feature loading
# ================================================================
def unwrap_feature(x):
    if x is None:
        return None
    while hasattr(x, "__len__") and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


# ================================================================
# Main
# ================================================================
def main():
    P = argparse.ArgumentParser()
    P.add_argument("--pairs-pkl", required=True)
    P.add_argument("--features-pkl", required=True)
    P.add_argument("--pca-model", required=True)
    P.add_argument(
        "--cov-ds",
        default=(
            "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
            "_no_null_returns_and_incomplete_descriptions_24k"
        ),
    )
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--l1-lambda", type=float, default=1e-5,
                    help="L1 penalty on attention weights (tune this)")
    P.add_argument("--epochs", type=int, default=30)
    P.add_argument("--batch-size", type=int, default=4096)
    P.add_argument("--train-sample", type=int, default=500_000,
                    help="Subsample training pairs for speed")
    P.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    P.add_argument("--theta-min", type=float, default=-4.5)
    P.add_argument("--theta-max", type=float, default=-1.0)
    P.add_argument("--theta-step", type=float, default=0.25)
    P.add_argument("--out-model", default="data/sparse_attention_model.pt")
    args = P.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Load pairs
    # ------------------------------------------------------------------
    print("Loading pairs...")
    pairs_df = pd.read_pickle(args.pairs_pkl)
    keep_cols = ["Company1", "Company2", "year", "correlation", "cosine_similarity"]
    pairs_df = pairs_df[[c for c in keep_cols if c in pairs_df.columns]]
    pairs_df["year"] = pairs_df["year"].astype(np.int16)
    pairs_df = pairs_df.dropna(subset=["correlation"])
    pairs_df["correlation"] = pairs_df["correlation"].astype(np.float32)
    pairs_df["cosine_similarity"] = pairs_df["cosine_similarity"].astype(np.float32)
    pairs_df["Company1"] = pairs_df["Company1"].astype(str)
    pairs_df["Company2"] = pairs_df["Company2"].astype(str)
    print(f"  {len(pairs_df)} pairs loaded")
    gc.collect()

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------
    all_years = sorted(pairs_df["year"].unique())
    split_idx = int(0.75 * len(all_years))
    train_years = all_years[:split_idx]
    test_years = all_years[split_idx:]
    print(f"  Train: {train_years[0]}-{train_years[-1]} ({len(train_years)} yrs)")
    print(f"  Test:  {test_years[0]}-{test_years[-1]}  ({len(test_years)} yrs)")

    # ------------------------------------------------------------------
    # Load features + PCA
    # ------------------------------------------------------------------
    print("Loading features + PCA...")
    df_f = pd.read_pickle(args.features_pkl)
    df_f["features"] = df_f["features"].apply(unwrap_feature)
    df_f = df_f.dropna(subset=["features"])
    df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

    df_c = load_dataset(args.cov_ds)["train"].to_pandas()
    df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
    df_feat = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
    df_feat = df_feat.dropna(subset=["sic_code"])
    df_feat["year"] = df_feat["year"].astype(int)
    df_feat = df_feat.reset_index(drop=True)

    pca_loaded = joblib.load(args.pca_model)
    if isinstance(pca_loaded, dict):
        pca = pca_loaded["pca"]
        feat_scaler = pca_loaded["scaler"]
    else:
        pca = pca_loaded
        feat_matrix_tmp = np.vstack(df_feat["features"].values)
        feat_scaler = StandardScaler().fit(feat_matrix_tmp)
        del feat_matrix_tmp
        gc.collect()

    components_T = pca.components_.T.astype(np.float32)
    dim = components_T.shape[1]
    n_rows = len(df_feat)
    pca_features = np.empty((n_rows, dim), dtype=np.float16)

    chunk_size = 1000
    print(f"  Projecting {n_rows} rows to {dim} PCA dims in chunks of {chunk_size}...")
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk_raw = np.vstack(df_feat["features"].values[start:end]).astype(np.float32)
        chunk_scaled = (chunk_raw - feat_scaler.mean_) / feat_scaler.scale_
        pca_features[start:end] = (chunk_scaled @ components_T).astype(np.float16)
        del chunk_raw, chunk_scaled

    del components_T
    gc.collect()
    print(f"  PCA features: {pca_features.shape}")

    # ------------------------------------------------------------------
    # Map pairs to PCA indices
    # ------------------------------------------------------------------
    print("Mapping pairs to feature indices...")
    df_feat["key"] = (
        df_feat["__index_level_0__"] + "_" + df_feat["year"].astype(str)
    )
    key_to_idx = dict(zip(df_feat["key"], range(len(df_feat))))

    pairs_df["idx1"] = (
        pairs_df["Company1"] + "_" + pairs_df["year"].astype(str)
    ).map(key_to_idx)
    pairs_df["idx2"] = (
        pairs_df["Company2"] + "_" + pairs_df["year"].astype(str)
    ).map(key_to_idx)
    n_before = len(pairs_df)
    pairs_df = pairs_df.dropna(subset=["idx1", "idx2"])
    pairs_df["idx1"] = pairs_df["idx1"].astype(int)
    pairs_df["idx2"] = pairs_df["idx2"].astype(int)
    print(f"  {len(pairs_df)} pairs matched (dropped {n_before - len(pairs_df)})")

    # ------------------------------------------------------------------
    # Spearman before training
    # ------------------------------------------------------------------
    rho_before, _ = spearmanr(
        pairs_df["cosine_similarity"].values, pairs_df["correlation"].values
    )
    print(f"\n  Spearman BEFORE training: {rho_before:.6f}")

    # ------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------
    train_mask = pairs_df["year"].isin(train_years)
    train_pairs = pairs_df[train_mask]
    if len(train_pairs) > args.train_sample:
        train_pairs = train_pairs.sample(n=args.train_sample, random_state=42)
    print(f"  Training on {len(train_pairs)} pairs")

    train_ds = PairIndexDataset(
        pca_features,
        train_pairs["idx1"].values,
        train_pairs["idx2"].values,
        train_pairs["correlation"].values,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(
        f"\nTraining SparseAttention  "
        f"(dim={dim}, lr={args.lr}, L1={args.l1_lambda}, epochs={args.epochs})"
    )
    model = SparseAttention(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for a, b, t in train_dl:
            a, b, t = a.to(device), b.to(device), t.to(device)
            pred = model(a, b)
            loss = mse_fn(pred, t) + args.l1_lambda * model.get_weights().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            w = model.get_weights().detach().cpu().numpy()
            sparsity = (w < 0.01).mean()
            print(
                f"  Epoch {epoch + 1:3d}  "
                f"loss={total_loss / n_batches:.6f}  "
                f"sparsity={sparsity:.1%}  "
                f"max_w={w.max():.4f}  "
                f"median_w={np.median(w):.4f}"
            )

    torch.save(model.state_dict(), args.out_model)
    print(f"  Saved model to {args.out_model}")

    # ------------------------------------------------------------------
    # Weight analysis
    # ------------------------------------------------------------------
    w = model.get_weights().detach().cpu().numpy()
    top_k = 20
    top_idx = np.argsort(w)[::-1][:top_k]
    print(f"\n  Top {top_k} PCA dimensions by learned weight:")
    for rank, idx in enumerate(top_idx):
        print(f"    {rank + 1:2d}. dim {idx:4d}  weight = {w[idx]:.4f}")
    print(f"  Dims with weight > 0.5:  {(w > 0.5).sum()} / {len(w)}")
    print(f"  Dims with weight > 0.1:  {(w > 0.1).sum()} / {len(w)}")
    print(f"  Dims with weight < 0.01: {(w < 0.01).sum()} / {len(w)}")

    # ------------------------------------------------------------------
    # Compute learned similarities for all pairs
    # ------------------------------------------------------------------
    print("\nComputing learned similarities...")
    model.eval()
    weighted_pca = pca_features * w[np.newaxis, :]
    norms = np.linalg.norm(weighted_pca, axis=1).clip(min=1e-10)

    all_idx1 = pairs_df["idx1"].values
    all_idx2 = pairs_df["idx2"].values
    learned_sims = np.empty(len(pairs_df), dtype=np.float32)

    for s in range(0, len(pairs_df), 500_000):
        e = min(s + 500_000, len(pairs_df))
        i1, i2 = all_idx1[s:e], all_idx2[s:e]
        learned_sims[s:e] = (
            (weighted_pca[i1] * weighted_pca[i2]).sum(1)
            / (norms[i1] * norms[i2])
        )

    pairs_df["learned_similarity"] = learned_sims

    # ------------------------------------------------------------------
    # Spearman after training
    # ------------------------------------------------------------------
    rho_after, _ = spearmanr(learned_sims, pairs_df["correlation"].values)
    print(f"  Spearman AFTER training:  {rho_after:.6f}")
    print(f"  Improvement: {rho_before:.6f} -> {rho_after:.6f}")

    # ------------------------------------------------------------------
    # Prepare distances for MST clustering
    # ------------------------------------------------------------------
    pairs_df["learned_distance"] = 1 - pairs_df["learned_similarity"]
    scaler_learned = StandardScaler()
    train_mask = pairs_df["year"].isin(train_years)
    scaler_learned.fit(pairs_df.loc[train_mask, ["learned_distance"]])
    pairs_df["learned_distance_scaled"] = scaler_learned.transform(
        pairs_df[["learned_distance"]]
    )

    # ------------------------------------------------------------------
    # Theta sweep (learned similarity) on train years
    # ------------------------------------------------------------------
    thetas = np.arange(
        args.theta_min, args.theta_max + args.theta_step / 2, args.theta_step
    )
    thetas = np.round(thetas, 2)
    print(f"\nSweeping {len(thetas)} thresholds (learned similarity)...")

    best_theta = None
    best_train_corr = -np.inf
    for theta in tqdm(thetas, desc="Theta sweep (learned)"):
        cdf = mst_cluster_years(
            pairs_df, train_years, theta, "learned_distance_scaled"
        )
        mc = calculate_weighted_avg_correlation(pairs_df, cdf)["avg_corr"].mean()
        if not np.isnan(mc) and mc > best_train_corr:
            best_train_corr = mc
            best_theta = theta

    print(f"  Best θ (learned) = {best_theta:.2f}  (train MC = {best_train_corr:.4f})")

    # ------------------------------------------------------------------
    # Evaluate learned similarity on test years
    # ------------------------------------------------------------------
    print(f"\nEvaluating learned similarity (θ={best_theta}) on test years...")
    test_cdf = mst_cluster_years(
        pairs_df, test_years, best_theta, "learned_distance_scaled"
    )
    test_mc = calculate_weighted_avg_correlation(pairs_df, test_cdf)["avg_corr"].mean()

    all_cdf = mst_cluster_years(
        pairs_df, all_years, best_theta, "learned_distance_scaled"
    )
    all_mc = calculate_weighted_avg_correlation(pairs_df, all_cdf)["avg_corr"].mean()

    # ------------------------------------------------------------------
    # Raw cosine MST baseline (for comparison)
    # ------------------------------------------------------------------
    print("Computing raw cosine MST baseline...")
    pairs_df["raw_distance"] = 1 - pairs_df["cosine_similarity"]
    scaler_raw = StandardScaler()
    scaler_raw.fit(
        pairs_df.loc[pairs_df["year"].isin(train_years), ["raw_distance"]]
    )
    pairs_df["raw_distance_scaled"] = scaler_raw.transform(
        pairs_df[["raw_distance"]]
    )

    best_raw_theta = None
    best_raw_corr = -np.inf
    for theta in tqdm(thetas, desc="Theta sweep (raw)"):
        cdf = mst_cluster_years(
            pairs_df, train_years, theta, "raw_distance_scaled"
        )
        mc = calculate_weighted_avg_correlation(pairs_df, cdf)["avg_corr"].mean()
        if not np.isnan(mc) and mc > best_raw_corr:
            best_raw_corr = mc
            best_raw_theta = theta

    raw_test_cdf = mst_cluster_years(
        pairs_df, test_years, best_raw_theta, "raw_distance_scaled"
    )
    raw_test_mc = calculate_weighted_avg_correlation(pairs_df, raw_test_cdf)[
        "avg_corr"
    ].mean()

    raw_all_cdf = mst_cluster_years(
        pairs_df, all_years, best_raw_theta, "raw_distance_scaled"
    )
    raw_all_mc = calculate_weighted_avg_correlation(pairs_df, raw_all_cdf)[
        "avg_corr"
    ].mean()

    # ------------------------------------------------------------------
    # SIC baseline
    # ------------------------------------------------------------------
    print("Computing SIC baseline...")
    df_comp = load_dataset(args.cov_ds)["train"].to_pandas()
    df_comp = df_comp[["cik", "year", "sic_code", "__index_level_0__"]].dropna(
        subset=["sic_code"]
    )
    df_comp["__index_level_0__"] = df_comp["__index_level_0__"].astype(str)
    df_comp["year"] = df_comp["year"].astype(int)

    sic_cluster_df = []
    for year in sorted(df_comp["year"].unique()):
        yd = df_comp[df_comp["year"] == year]
        clusters = yd.groupby("sic_code")["__index_level_0__"].apply(list).to_dict()
        sic_cluster_df.append({"year": year, "clusters": clusters})
    sic_cluster_df = pd.DataFrame(sic_cluster_df)
    sic_cluster_df["year"] = sic_cluster_df["year"].astype(int)

    sic_test = sic_cluster_df[sic_cluster_df["year"].isin(test_years)]
    sic_test_mc = calculate_weighted_avg_correlation(pairs_df, sic_test)[
        "avg_corr"
    ].mean()
    sic_all_mc = calculate_weighted_avg_correlation(pairs_df, sic_cluster_df)[
        "avg_corr"
    ].mean()

    pop_corr = pairs_df["correlation"].mean()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS  (pair-weighted evaluation)")
    print("=" * 72)
    print(f"  Spearman rho (raw cosine):       {rho_before:.6f}")
    print(f"  Spearman rho (learned attention): {rho_after:.6f}")
    print()
    print(f"  {'Method':35s} {'OOS (test)':>12s} {'All years':>12s}")
    print(f"  {'-' * 35} {'-' * 12} {'-' * 12}")
    print(
        f"  {'Sparse Attention + MST':35s} {test_mc:>12.4f} {all_mc:>12.4f}"
    )
    print(
        f"  {'Raw Cosine + MST':35s} {raw_test_mc:>12.4f} {raw_all_mc:>12.4f}"
    )
    print(
        f"  {'SIC code clusters':35s} {sic_test_mc:>12.4f} {sic_all_mc:>12.4f}"
    )
    print(
        f"  {'Population mean':35s} {pop_corr:>12.4f} {pop_corr:>12.4f}"
    )
    print("=" * 72)
    print(
        f"\n  Active dimensions (weight > 0.1): "
        f"{(w > 0.1).sum()} / {len(w)}"
    )
    print(f"  Sparse dimensions (weight < 0.01): {(w < 0.01).sum()} / {len(w)}")


if __name__ == "__main__":
    main()
