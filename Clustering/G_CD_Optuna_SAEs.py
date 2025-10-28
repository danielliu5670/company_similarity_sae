from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
import psutil
import os
import joblib
import optuna
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import json
from sklearn.preprocessing import StandardScaler
import json
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import networkx as nx
from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
import optuna
from functools import partial
from joblib import Parallel, delayed
import logging
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import psutil
import os
process = psutil.Process(os.getpid())
print(f"Memory usage before: {process.memory_info().rss / 1024 ** 2:.2f} MB")

ds_compinfo = load_dataset("Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k")
df_compinfo = ds_compinfo['train'].to_pandas()
df_compinfo = df_compinfo[["cik", "year", "sic_code", "ticker", "__index_level_0__"]]
df_compinfo = df_compinfo.dropna(subset=['sic_code'])
print(f"Number of rows after dropping missing sic_code: {len(df_compinfo)}")
# Define a function to classify SIC codes into industries based on the first two digits
def classify_sic(sic_code):
    # Extract the first two digits of the SIC code
    first_two_digits = int(str(sic_code)[:2])

    # Map to industry categories
    if 1 <= first_two_digits <= 9:
        return 'Agriculture, Forestry, And Fishing'
    elif 10 <= first_two_digits <= 14:
        return 'Mining'
    elif 15 <= first_two_digits <= 17:
        return 'Construction'
    elif 20 <= first_two_digits <= 39:
        return 'Manufacturing'
    elif 40 <= first_two_digits <= 49:
        return 'Transportation, Communications, Electric, Gas, And Sanitary Services'
    elif 50 <= first_two_digits <= 51:
        return 'Wholesale Trade'
    elif 52 <= first_two_digits <= 59:
        return 'Retail Trade'
    elif 60 <= first_two_digits <= 67:
        return 'Finance, Insurance, And Real Estate'
    elif 70 <= first_two_digits <= 89:
        return 'Services'
    elif 90 <= first_two_digits <= 99:
        return 'Public Administration'
    else:
        return 'Unknown'
ds_compinfo = 0
# Apply the classification to the SIC codes in the dataset
df_compinfo['industry_classification'] = df_compinfo['sic_code'].apply(classify_sic)
process = psutil.Process(os.getpid())
print(f"Memory usage before: {process.memory_info().rss / 1024 ** 2:.2f} MB")

pairs_ds = load_dataset("v1ctor10/cos_sim_4000pca_exp")
pairs_df = pairs_ds['train'].to_pandas()

pairs_df = pairs_df.dropna(subset=['correlation']).reset_index(drop=True)
pairs_df["year"] = pairs_df["year"].astype(int)

def calculate_avg_correlation(TO_ANALYSE_DF, cluster_df, cluster_type):
    avg_correlations = []

    for _, row in tqdm(cluster_df.iterrows(), desc=f"Calculating Stats for {cluster_type}", total=len(cluster_df)):
        year = row['year']
        clusters = row['clusters']
        year_data = TO_ANALYSE_DF[TO_ANALYSE_DF['year'] == year]
        cluster_stats = []
        for cluster_id, companies in clusters.items():
            if len(companies) <= 1:  # Skip clusters with only 1 company
                continue

            # Get all pairs of companies within the cluster
            cluster_pairs = year_data[
                (year_data['Company1'].isin(companies) & year_data['Company2'].isin(companies))
            ]
            # Calculate statistics for the cluster
            if not cluster_pairs.empty:
                try:
                    correlations = cluster_pairs['correlation']
                except KeyError:
                    # Handle alternative column name if 'correlation' doesn't exist
                    correlations = cluster_pairs['ActualCorrelation']
                cluster_stats.append(correlations.mean())

        # Aggregate statistics across clusters for the year
        if cluster_stats:
            avg_correlations.append({
                'year': year,
                f'{cluster_type}AvgCorrelation': sum(cluster_stats) / len(cluster_stats)
            })
        else:
            avg_correlations.append({
                'year': year,
                f'{cluster_type}AvgCorrelation': np.nan
            })

    return pd.DataFrame(avg_correlations)


def create_cluster_dfs(df_compinfo, year_cluster_df):
    """
    Create SIC and Industry cluster DataFrames for each year.
    """
    year_SIC_cluster_df = []
    year_Industry_cluster_df = []

    for year in tqdm(sorted(df_compinfo['year'].unique()), desc="Generating Cluster DataFrames"):
        # Filter companies for the year
        year_data = df_compinfo[df_compinfo['year'] == year]

        # SIC clusters
        sic_clusters = year_data.groupby('sic_code')['__index_level_0__'].apply(list).to_dict()
        year_SIC_cluster_df.append({'year': year, 'clusters': sic_clusters})

        # Industry clusters
        industry_clusters = year_data.groupby('industry_classification')['__index_level_0__'].apply(list).to_dict()
        year_Industry_cluster_df.append({'year': year, 'clusters': industry_clusters})

    return pd.DataFrame(year_SIC_cluster_df), pd.DataFrame(year_Industry_cluster_df)

year_SIC_cluster_df, year_Industry_cluster_df = create_cluster_dfs(df_compinfo, pairs_df)
year_SIC_cluster_df["year"] = year_SIC_cluster_df["year"].astype(int)
year_Industry_cluster_df["year"] = year_Industry_cluster_df["year"].astype(int)
year_SIC_cluster_df = year_SIC_cluster_df.sort_values(by='year').reset_index(drop=True)
year_Industry_cluster_df = year_Industry_cluster_df.sort_values(by='year').reset_index(drop=True)

# Create directory if it doesn't exist
os.makedirs("./data/Final Results", exist_ok=True)
year_SIC_cluster_df.to_pickle("./data/Final Results/year_cluster_dfSIC.pkl")
year_Industry_cluster_df.to_pickle("./data/Final Results/year_cluster_dfINDUSTRY.pkl")


global sic_avg_corr
global industry_avg_corr
sic_avg_corr = calculate_avg_correlation(pairs_df, year_SIC_cluster_df, "SIC")
industry_avg_corr = calculate_avg_correlation(pairs_df, year_Industry_cluster_df, "Industry")

sic_avg_corr.to_csv("./data/sic_avg_corr.csv", index=False)
industry_avg_corr.to_csv("./data/industry_avg_corr.csv", index=False)

sic_p = sic_avg_corr.mean()[1]
ind_p = industry_avg_corr.mean()[1]
pop_p = pairs_df["correlation"].mean()
print(f"sic: {sic_p}, industry: {ind_p}, population: {pop_p}")

# Prepping distance metric
pairs_df['cosine_distance'] = 1 - pairs_df['cosine_similarity']

scaler = StandardScaler()
pairs_df['cosine_distance_scaled'] = scaler.fit_transform(pairs_df[['cosine_distance']])


# Helper functions:

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    filename='optuna_optimization.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def perform_clustering_per_year(
    TO_ANALYSE_DF,
    years_to_cluster,
    threshold,
    linkage_method='single'
):
    """
    Perform clustering on specified years using MST thresholding.

    Parameters:
    - pairs_df (pd.DataFrame): DataFrame containing company pairs with 'Company1', 'Company2', 'year', and 'sum_abs_diff_scaled_01'.
    - years_to_cluster (list): List of years to perform clustering on.
    - threshold (float): Threshold for forming clusters by removing edges from the MST.
    - linkage_method (str): Linkage method for hierarchical clustering ('single', 'complete', 'average', 'ward').

    Returns:
    - pd.DataFrame: DataFrame with 'year' and 'clusters' columns.
    """

    # Filter the DataFrame for the specified years
    filtered_df = TO_ANALYSE_DF[TO_ANALYSE_DF['year'].isin(years_to_cluster)]

    # Get sorted list of unique years within the specified subset
    unique_years = sorted(filtered_df['year'].unique())

    # Initialize list to collect clustering results
    clustering_results = []

    # Iterate over each year with a progress bar
    for year in tqdm(unique_years, desc=f"Clustering Years {years_to_cluster}"):
        # Filter data for the current year
        year_df = filtered_df[filtered_df['year'] == year]

        # Check if there are enough company pairs to form clusters
        if year_df.empty:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Initialize an empty undirected graph
        G = nx.Graph()

        # Add edges to the graph with 'sum_abs_diff_scaled_01' as the weight
        edges = list(zip(year_df['Company1'], year_df['Company2'], year_df['cosine_distance_scaled']))
        G.add_weighted_edges_from(edges)

        # Check if the graph has at least one edge
        if G.number_of_edges() == 0:
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Compute the Minimum Spanning Tree (MST)
        try:
            mst = nx.minimum_spanning_tree(G, weight='weight')
        except Exception as e:
            print(f"Error computing MST for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Threshold the MST: remove edges with weight > threshold
        try:
            edges_to_remove = [(u, v) for u, v, d in mst.edges(data=True) if d['weight'] > threshold]
            mst.remove_edges_from(edges_to_remove)
        except Exception as e:
            print(f"Error thresholding MST for year {year}: {e}")
            clustering_results.append({'year': year, 'clusters': {}})
            continue

        # Find connected components (clusters) in the thresholded MST
        clusters = list(nx.connected_components(mst))

        # Assign unique cluster IDs
        cluster_dict = {}
        for idx, cluster in enumerate(clusters, start=1):
            cluster_dict[idx] = sorted(list(cluster))

        # Append the result
        clustering_results.append({'year': year, 'clusters': cluster_dict})

    # Convert the results to a DataFrame
    result_df = pd.DataFrame(clustering_results)

    return result_df


def optimise_cluster_parameters(TO_ANALYSE_DF, pairs_df, use_holdout=True):
    """
    Optimize clustering parameters using Optuna with temporal cross-validation.

    If use_holdout=True: Uses only first 75% of data for optimization,
                         keeping last 25% for final out-of-sample testing.
    If use_holdout=False: Uses all data for optimization (original behavior).
    """
    # Get all unique years
    all_years = sorted(pairs_df['year'].unique())
    n_total_years = len(all_years)

    # Define splits for proper out-of-sample evaluation
    if use_holdout:
        # Split: 25% (A), 50% (B), 25% (C held out)
        split_A_end = int(0.25 * n_total_years)
        split_B_end = int(0.75 * n_total_years)

        years_A = all_years[:split_A_end]              # First 25%
        years_B = all_years[split_A_end:split_B_end]   # Middle 50%
        years_C = all_years[split_B_end:]              # Last 25% (HELD OUT)

        # Only use A+B for optimization
        optimization_years = years_A + years_B

        print(f"\n=== Data Split for Out-of-Sample Evaluation ===")
        print(f"Period A (25%): {years_A[0]}-{years_A[-1]} ({len(years_A)} years)")
        print(f"Period B (50%): {years_B[0]}-{years_B[-1]} ({len(years_B)} years)")
        print(f"Period C (25%, HELD OUT): {years_C[0]}-{years_C[-1]} ({len(years_C)} years)")
        print(f"Optimization will use only periods A and B: {optimization_years[0]}-{optimization_years[-1]}")
        print("=" * 50 + "\n")

        # Filter data to only include optimization years
        optimization_df = pairs_df[pairs_df['year'].isin(optimization_years)]
    else:
        # Use all data (original behavior)
        optimization_df = pairs_df
        optimization_years = all_years
        years_A = None
        years_B = None
        years_C = None

    def objective(trial):
        # Suggest values for hyperparameters
        threshold = trial.suggest_float('threshold', -3.53, -3.3, step=0.002)
        linkage_method = 'single'

        if use_holdout:
            # Simple evaluation on Period A and Period B separately
            
            # Evaluate on Period A (25% of total data)
            period_A_df = optimization_df[optimization_df['year'].isin(years_A)]
            cluster_df_A = perform_clustering_per_year(
                TO_ANALYSE_DF=period_A_df,
                years_to_cluster=years_A,
                threshold=threshold,
                linkage_method=linkage_method
            )
            avg_corr_A = calculate_avg_correlation(
                TO_ANALYSE_DF=period_A_df,
                cluster_df=cluster_df_A,
                cluster_type="PeriodA"
            )
            score_A = avg_corr_A['PeriodAAvgCorrelation'].mean()
            
            # Evaluate on Period B (50% of total data)  
            period_B_df = optimization_df[optimization_df['year'].isin(years_B)]
            cluster_df_B = perform_clustering_per_year(
                TO_ANALYSE_DF=period_B_df,
                years_to_cluster=years_B,
                threshold=threshold,
                linkage_method=linkage_method
            )
            avg_corr_B = calculate_avg_correlation(
                TO_ANALYSE_DF=period_B_df,
                cluster_df=cluster_df_B,
                cluster_type="PeriodB"
            )
            score_B = avg_corr_B['PeriodBAvgCorrelation'].mean()
            
            # Average across both periods (as stated in paper)
            overall_avg_corr = (score_A + score_B) / 2
            
            # Handle NaN cases
            if np.isnan(overall_avg_corr):
                overall_avg_corr = -np.inf
                
            # Log the parameters and the resulting correlation
            logging.info(f"Threshold: {threshold}, Score A: {score_A:.4f}, Score B: {score_B:.4f}, Average: {overall_avg_corr:.4f}")
            
        else:
            # Original behavior - evaluate on all optimization data
            cluster_df = perform_clustering_per_year(
                TO_ANALYSE_DF=optimization_df,
                years_to_cluster=optimization_years,
                threshold=threshold,
                linkage_method=linkage_method
            )
            avg_corr_df = calculate_avg_correlation(
                TO_ANALYSE_DF=optimization_df,
                cluster_df=cluster_df,
                cluster_type="All"
            )
            overall_avg_corr = avg_corr_df['AllAvgCorrelation'].mean()
            
            if np.isnan(overall_avg_corr):
                overall_avg_corr = -np.inf
                
            logging.info(f"Threshold: {threshold}, Average Correlation: {overall_avg_corr:.4f}")

        return overall_avg_corr  # Optuna will maximize this

    # Create an Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=1))

    # Optimize the study with temporal cross-validation
    study.optimize(objective, n_trials=150, timeout=28800, callbacks=[save_study_callback])

    # Retrieve the best parameters
    best_params = study.best_params
    best_score = study.best_value

    print(f"\nBest parameters found by Optuna (using {optimization_years[0]}-{optimization_years[-1]}):")
    print(f"Threshold: {best_params['threshold']}")
    print(f"Linkage Method: {best_params.get('linkage_method', 'single')}")
    print(f"Best Average Correlation (on optimization data): {best_score}")

    # If using holdout, evaluate on held-out test set
    if use_holdout:
        print("\n" + "=" * 50)
        print("Evaluating on held-out test set (Period C)...")
        print("=" * 50)

        # Filter data for test years
        test_df = pairs_df[pairs_df['year'].isin(years_C)]

        # Apply best threshold to test years
        test_cluster_df = perform_clustering_per_year(
            TO_ANALYSE_DF=test_df,
            years_to_cluster=years_C,
            threshold=best_params['threshold'],
            linkage_method=best_params.get('linkage_method', 'single')
        )

        # Calculate average correlation for test set
        test_avg_corr_df = calculate_avg_correlation(
            TO_ANALYSE_DF=test_df,
            cluster_df=test_cluster_df,
            cluster_type="Test"
        )

        test_score = test_avg_corr_df['TestAvgCorrelation'].mean()

        # Calculate baselines for test period
        test_sic = sic_avg_corr[sic_avg_corr['year'].isin(years_C)]['SICAvgCorrelation'].mean()
        test_industry = industry_avg_corr[industry_avg_corr['year'].isin(years_C)]['IndustryAvgCorrelation'].mean()
        test_population = test_df['correlation'].mean()

        print(f"\n=== OUT-OF-SAMPLE TEST RESULTS ({years_C[0]}-{years_C[-1]}) ===")
        print(f"MST Clustering (θ={best_params['threshold']:.3f}): {test_score:.4f}")
        print(f"SIC Baseline: {test_sic:.4f}")
        print(f"Industry Baseline: {test_industry:.4f}")
        print(f"Population Mean: {test_population:.4f}")
        print("=" * 50)

        # Save test results
        test_results = {
            'optimization_years': f"{optimization_years[0]}-{optimization_years[-1]}",
            'test_years': f"{years_C[0]}-{years_C[-1]}",
            'best_threshold': best_params['threshold'],
            'optimization_score': best_score,
            'test_score': test_score,
            'test_sic_baseline': test_sic,
            'test_industry_baseline': test_industry,
            'test_population_mean': test_population
        }

        # Save to file
        with open('./data/out_of_sample_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)

        print(f"\nTest results saved to ./data/out_of_sample_test_results.json")

        # Add test results to best_params for return
        best_params['test_score'] = test_score
        best_params['test_years'] = years_C

    return study, best_params


def save_study_callback(study, trial):
    """
    Callback to save trial results to CSV after each trial completes.
    """
    trial_data = {
        'number': trial.number,
        'value': trial.value,
        'state': str(trial.state)
    }

    for k, v in trial.params.items():
        trial_data[k] = v

    df = pd.DataFrame([trial_data])
    os.makedirs("./data/Clustering Optuna Study/", exist_ok=True)
    filename = f"./data/Clustering Optuna Study/{cluster_name}-cluster_study_results.csv"
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, index=False, mode='a', header=not file_exists)
    print("Trial added to CSV.")
    print(df)
    print("\n")

global cluster_name
cluster_name = "C-CD" # C-CD = Cosine Distance
TO_ANALYSE_DF = pairs_df

# Optimize clustering parameters with proper out-of-sample evaluation
study, best_params = optimise_cluster_parameters(
    TO_ANALYSE_DF=TO_ANALYSE_DF,
    pairs_df=pairs_df,
    use_holdout=True  # Set to True for proper out-of-sample evaluation
)

# Load and visualize study results
filename = f"./data/Clustering Optuna Study/{cluster_name}-cluster_study_results.csv"
df_study = pd.read_csv(filename)
df_study_sorted = df_study.sort_values("value", ascending=False)
print("\nTop 5 trials by optimization score:")
print(df_study_sorted.head())

# Create visualization
plt.figure(figsize=(10, 6))
plt.bar(df_study['threshold'], df_study['value'], width=0.01, align='center', alpha=0.75)
plt.title("Histogram of Threshold vs Value (Optimization Data Only)", fontsize=14)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Value (mean correlation)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add vertical line for best threshold
plt.axvline(x=best_params['threshold'], color='red', linestyle='--',
            label=f'Best θ={best_params["threshold"]:.3f}')
plt.legend()

# Save the plot
os.makedirs("./images/", exist_ok=True)
plt.savefig('./images/optuna_study.png', dpi=300, bbox_inches='tight')
plt.show()

# If test score exists, create comparison plot
if 'test_score' in best_params:
    plt.figure(figsize=(10, 6))

    # Prepare data for comparison
    methods = ['MST\nClustering', 'SIC\nBaseline', 'Industry\nBaseline', 'Population\nMean']

    # Get test scores
    test_years = best_params['test_years']
    test_df = pairs_df[pairs_df['year'].isin(test_years)]
    test_sic = sic_avg_corr[sic_avg_corr['year'].isin(test_years)]['SICAvgCorrelation'].mean()
    test_industry = industry_avg_corr[industry_avg_corr['year'].isin(test_years)]['IndustryAvgCorrelation'].mean()
    test_population = test_df['correlation'].mean()

    scores = [best_params['test_score'], test_sic, test_industry, test_population]

    # Create bar plot
    bars = plt.bar(methods, scores, alpha=0.75)
    bars[0].set_color('green')  # MST Clustering
    bars[1].set_color('blue')   # SIC
    bars[2].set_color('orange') # Industry
    bars[3].set_color('gray')   # Population

    plt.title(f'Out-of-Sample Test Performance ({test_years[0]}-{test_years[-1]})', fontsize=14)
    plt.ylabel('Average Correlation', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('./images/out_of_sample_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\nOptimization complete! Check ./data/out_of_sample_test_results.json for detailed results.")
