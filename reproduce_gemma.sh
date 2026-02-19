#!/bin/bash
set -e

python extract_features_gemma.py \
    --model google/gemma-3-4b-pt \
    --sae-repo google/gemma-scope-2-4b-pt \
    --sae-path resid_post/layer_22_width_65k_l0_medium \
    --layer 29 \
    --output data/gemma_features.pkl

python compute_similarities_gemma.py \
    --features-pkl data/gemma_features.pkl \
    --pca-components 4000 \
    --out-pairs data/gemma_pairs.pkl \
    --out-pca data/gemma_pca_model.pkl

python cluster_gemma.py \
    --pairs-pkl data/gemma_pairs.pkl \
    --threshold -2.7 \
    --features-pkl data/gemma_features.pkl \
    --pca-model data/gemma_pca_model.pkl \
    --out-clusters data/gemma_year_cluster_dfC-CD.pkl \
    --out-clusters-kmeans data/gemma_year_cluster_dfC-CD_kmeans.pkl

strong_pca_path=data/gemma_strong_features_pca.pkl

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    cluster_feature_gpu.py \
    --clusters-pkl data/gemma_year_cluster_dfC-CD.pkl \
    --local-features data/gemma_features.pkl \
    --local-pairs data/gemma_pairs.pkl \
    --pca-model data/gemma_pca_model.pkl \
    --num-candidates 1000 \
    --out $strong_pca_path

strong_kmeans_path=data/gemma_strong_features_kmeans.pkl

python strong_features_kmeans.py \
    --clusters-pkl data/gemma_year_cluster_dfC-CD.pkl \
    --local-features data/gemma_features.pkl \
    --pca-model data/gemma_pca_model.pkl \
    --num-candidates 1000 \
    --n-clusters 5 \
    --out $strong_kmeans_path

python strong_features_proportion.py \
    --strong_features_path $strong_pca_path \
    --out_image_path images/gemma_features_proportion_pca.png

python strong_features_proportion.py \
    --strong_features_path $strong_kmeans_path \
    --out_image_path images/gemma_features_proportion_kmeans.png

# python distribution_summed_sae_features.py \
#     --features-pkl data/gemma_features.pkl \
#     --img_path images/gemma_distribution.png
