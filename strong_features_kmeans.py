#!/usr/bin/env python3
# strong_features_kmeans.py
"""
GPU-parallel "strong latent" miner for clusters using k-means.

Examples
────────

# 1 ▸ replay previous run
torchrun --standalone --nproc_per_node 8 \
    strong_features_kmeans.py \
    --clusters-pkl year_cluster_dfC-CD.pkl \
    --scores-folder ../fuzzing_scores \
    --reference-pkl ../data/PCA_strong_features_clusters_random_subset.pkl \
    --out strong_features_clusters_random_subset.pkl

# 2 ▸ fresh random subset of 200 clusters
torchrun --standalone --nproc_per_node 8 \
    strong_features_kmeans.py \
    --clusters-pkl Clustering/data/Final Results/year_cluster_dfC-CD.pkl \
    --scores-folder ../fuzzing_scores \
    --sample-size 200 \
    --out strong_features_new_random_200.pkl
"""



# ───────────────────────── imports ─────────────────────────
import os, random, argparse, pickle, warnings, re
import numpy as np, pandas as pd
import torch, torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib, datasets, pickle  # noqa: F401  (pickle re-import for type checkers)

torch.backends.cuda.matmul.allow_tf32 = True  # MI250-X likes TF32

# ───────────────────────── helpers ─────────────────────────
_latent_re = re.compile(r"latent(\d+)", re.I)

def extract_latent(fn: str) -> int | None:
    m = _latent_re.search(fn)
    return int(m.group(1)) if m else None

def load_scored_latents(folder: str, n: int = 1000, seed: int = 42) -> np.ndarray:
    """Pick `n` unique latent indices that have score files in `folder`."""
    lat = {extract_latent(f) for f in os.listdir(folder) if f.endswith(".txt")}
    lat.discard(None)
    rng = np.random.default_rng(seed)
    return rng.choice(sorted(lat), size=min(n, len(lat)), replace=False).astype(int)

def compute_cluster_coherence(Z: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    """
    Compute average within-cluster cosine similarity.
    Returns a scalar measuring cluster coherence.
    """
    coherence_sum = 0.0
    count = 0
    
    for label in np.unique(labels):
        mask = labels == label
        cluster_points = Z[mask]
        n_points = cluster_points.shape[0]
        
        if n_points > 1:
            # Normalize for cosine similarity
            normalized = F.normalize(cluster_points, p=2, dim=1)
            # Compute pairwise similarities within cluster
            sim_matrix = normalized @ normalized.T
            # Sum upper triangle (excluding diagonal)
            coherence_sum += (sim_matrix.sum() - n_points) / 2
            count += n_points * (n_points - 1) / 2
    
    return coherence_sum / count if count > 0 else torch.tensor(0.0)

# ───────────────────────── CLI ─────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--clusters-pkl", required=True,
              help="pickle with yearly cluster dicts (e.g. year_cluster_dfC-CD.pkl)")
grp = P.add_mutually_exclusive_group()
grp.add_argument("--reference-pkl",
                 help="previous output pickle – process only those clusters")
grp.add_argument("--sample-size", type=int,
                 help="fresh random subset of this many clusters")
P.add_argument("--feature-ds",default="marco-molinari/company_reports_with_features")
P.add_argument("--cov-ds",default=("Mateusz1017/annual_reports_tokenized_llama3_logged_returns_"
                                   "no_null_returns_and_incomplete_descriptions_24k"))
P.add_argument("--pca-model",  default="../data/global_pca_model.pkl")
P.add_argument("--scores-folder", default="./fuz_scores")
P.add_argument("--n-clusters", type=int, default=5,
              help="number of k-means clusters (default: 5)")
P.add_argument("--out", default="strong_features_clusters_random_subset.pkl")
args = P.parse_args()

# ───── select clusters ─────
with open(args.clusters_pkl, "rb") as f:
    clusters_df = pickle.load(f)

all_clusters = [c for row in clusters_df["clusters"]
                  for c in row.values() if len(c) > 1]

if args.reference_pkl:
    with open(args.reference_pkl, "rb") as f:
        ref = pickle.load(f)
    ref_set = {frozenset(map(int, k.split(","))) for k in ref}
    all_clusters = [c for c in all_clusters if frozenset(c) in ref_set]
    print(f"[info] {len(all_clusters)} clusters taken from reference pickle")
elif args.sample_size:
    random.seed(42); random.shuffle(all_clusters)
    all_clusters = all_clusters[:args.sample_size]
    print(f"[info] random sample of {len(all_clusters)} clusters")
else:
    print(f"[info] processing every cluster in file ({len(all_clusters)})")

# ───────────────────────── DDP ─────────────────────────
_DDP = "RANK" in os.environ
if _DDP:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    device      = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
else:
    rank, world = 0, 1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
warnings.filterwarnings("ignore")

if _DDP and rank == 0:
    for _ in tqdm(range(60), desc="⏳ init", total=60, position=0):
        torch.cuda._sleep(int(1e6))
if _DDP:
    dist.barrier()

# ───────────────────────── load data & PCA ─────────────────────────
df_f = datasets.load_dataset(args.feature_ds)["train"].to_pandas()
df_c = datasets.load_dataset(args.cov_ds)["train"].to_pandas()
df   = (pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
          .dropna(subset=["sic_code", "features"]))
df["features"] = df["features"].apply(lambda x: np.asarray(x[0], dtype=np.float32))

feat_matrix = np.vstack(df["features"].values)
scaler = StandardScaler().fit(feat_matrix)

if os.path.exists(args.pca_model):
    pca = joblib.load(args.pca_model)
else:
    n_comp = min(4000, *feat_matrix.shape)
    pca = PCA(n_components=n_comp).fit(scaler.transform(feat_matrix))
    os.makedirs(os.path.dirname(args.pca_model) or ".", exist_ok=True)
    joblib.dump(pca, args.pca_model)
    if rank == 0:
        print(f"[info] fitted and saved PCA ({n_comp} components) to {args.pca_model}")

mean_t  = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device)
scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
W_t     = torch.tensor(pca.components_.T, dtype=torch.float32, device=device)  # (D,P)

cand_idx   = load_scored_latents(args.scores_folder, 1000)
vec_dim    = df["features"].iat[0].shape[0]

# ───────────────────────── main loop ─────────────────────────
my_clusters = [c for i, c in enumerate(all_clusters) if i % world == rank]
pbar = tqdm(my_clusters, desc=f"rank{rank}", position=rank, leave=False)

local_out = {}
for firms in pbar:
    firms = set(firms)
    
    # Get all company data for this cluster
    mask_rows = df["__index_level_0__"].isin(firms)
    if mask_rows.sum() < args.n_clusters:
        continue  # Skip if not enough companies for clustering
    
    raw_np = np.vstack(df.loc[mask_rows, "features"].values).astype(np.float32)
    
    X_scaled = (torch.tensor(raw_np, device=device) - mean_t) / scale_t       # (N,D)
    Z        = X_scaled @ W_t                                                # (N,P)

    # Perform k-means clustering on CPU (sklearn is CPU-based)
    Z_cpu = Z.cpu().numpy()
    kmeans = KMeans(n_clusters=min(args.n_clusters, len(Z_cpu)), 
                    random_state=42, n_init=10)
    labels = kmeans.fit_predict(Z_cpu)
    
    # Compute baseline cluster coherence
    baseline_coherence = compute_cluster_coherence(Z, labels)

    # ── Impact score for each candidate latent ──
    idx_to_impact = {}
    for j in cand_idx:
        if j >= vec_dim:
            continue
        raw_j  = torch.tensor(raw_np[:, j], device=device)
        delta  = (-(raw_j / scale_t[j])).unsqueeze(1) * W_t[j]               # (N,P)
        Z_perturbed = Z + delta
        
        # Re-cluster with perturbed features
        Z_pert_cpu = Z_perturbed.cpu().numpy()
        kmeans_pert = KMeans(n_clusters=min(args.n_clusters, len(Z_pert_cpu)), 
                            random_state=42, n_init=10)
        labels_pert = kmeans_pert.fit_predict(Z_pert_cpu)
        
        # Compute new coherence
        new_coherence = compute_cluster_coherence(Z_perturbed, labels_pert)
        
        # Impact = decrease in coherence (higher = more important feature)
        impact = (baseline_coherence - new_coherence).item()
        idx_to_impact[j] = impact

    res = pd.Series(idx_to_impact, name="impact").to_frame()

    # ── Robust adaptive elbow search ──
    quantiles = (0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50)
    strong_idx = weak_idx = None

    def impact_with_set(idx_list):
        if len(idx_list) == 0:
            return 0.0
        delta = torch.zeros_like(Z)
        for j in idx_list:
            raw_j = torch.tensor(raw_np[:, j], device=device)
            delta += (-(raw_j / scale_t[j])).unsqueeze(1) * W_t[j]
        Z_perturbed = Z + delta
        Z_pert_cpu = Z_perturbed.cpu().numpy()
        kmeans_pert = KMeans(n_clusters=min(args.n_clusters, len(Z_pert_cpu)), 
                            random_state=42, n_init=10)
        labels_pert = kmeans_pert.fit_predict(Z_pert_cpu)
        new_coherence = compute_cluster_coherence(Z_perturbed, labels_pert)
        return (baseline_coherence - new_coherence).item()

    for q in quantiles:
        thr        = res["impact"].quantile(q)
        strong_idx = res[res["impact"] >  thr].index.to_numpy(int)
        weak_idx   = res[res["impact"] <= thr].index.to_numpy(int)
        if len(strong_idx) == 0 or len(weak_idx) == 0:
            continue
        impact_strong = impact_with_set(strong_idx)
        impact_weak   = impact_with_set(weak_idx)

        if impact_weak == 0:
            if impact_strong == 0:   # both zero – meaningless, test next q
                continue
            break                 # weak 0 & strong >0  → good elbow
        if impact_strong / impact_weak > 1:
            break                 # standard elbow condition
    else:
        # fallback: fixed 90-th percentile
        thr        = res["impact"].quantile(0.90)
        strong_idx = res[res["impact"] > thr].index.to_numpy(int)
        weak_idx   = res[res["impact"] <= thr].index.to_numpy(int)

    key = ",".join(sorted(map(str, firms)))
    local_out[key] = {
        "strong_features": strong_idx,
        "weak_features":   weak_idx,
        "impact_by_feature":  res,
        "baseline_coherence": baseline_coherence.item()
    }

# ───────────────────────── gather & save ─────────────────────────
if _DDP:
    gather = [None] * world
    dist.gather_object(local_out, gather if rank == 0 else None, dst=0)
else:
    gather = [local_out]

if rank == 0:
    merged = {}
    for d in gather:
        merged.update(d)
    with open(args.out, "wb") as f:
        pickle.dump(merged, f)
    print(f"\n✅ wrote '{args.out}' with {len(merged)} clusters.")

if _DDP:
    dist.barrier()
    dist.destroy_process_group()

