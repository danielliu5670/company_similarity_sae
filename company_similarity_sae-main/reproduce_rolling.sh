strong_features_path=data/rolling_strong_features_clusters_random_subset.pkl

python3 cluster_feature_gpu.py     --clusters-pkl data/rolling_year_cluster_dfCD.pkl     --out $strong_features_path --reference-pkl $strong_features_path  --pca-model data/global_pca_model_4000.pkl

python3 interp_over_sparsity.py --strong_features_path $strong_features_path --top_1_percent_clusters 25 --out_image_path images/rolling_interp_over_sparsity.png

python3 strong_features_proportion.py --strong_features_path $strong_features_path --out_image_path images/rolling_features_proportion.png