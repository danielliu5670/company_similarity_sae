#!/usr/bin/env python3

import os
import pandas as pd
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def split_by_year(df, year_col='year', train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    years = sorted(df[year_col].unique())
    n_years = len(years)
    train_end = int(n_years * train_ratio)
    val_end = int(n_years * (train_ratio + val_ratio))
    train_years, val_years, test_years = years[:train_end], years[train_end:val_end], years[val_end:]
    return (
        df[df[year_col].isin(train_years)],
        df[df[year_col].isin(val_years)],
        df[df[year_col].isin(test_years)]
    )

def save_splits(train_df, val_df, test_df, name, save_csv=True):
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        pkl_path = os.path.join(OUTPUT_DIR, f"{name}_{split_name}.pkl")
        split_df.to_pickle(pkl_path)
        log(f"Saved {pkl_path}")
        if save_csv:
            csv_path = os.path.join(OUTPUT_DIR, f"{name}_{split_name}.csv")
            split_df.to_csv(csv_path, index=False)
            log(f"Saved {csv_path}")

log("Loading marco-molinari/company_reports_with_features")
ds_features = load_dataset("marco-molinari/company_reports_with_features")
df_features = ds_features['train'].to_pandas()

log("Loading Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k")
ds_compinfo = load_dataset("Mateusz1017/annual_reports_tokenized_llama3_logged_returns_no_null_returns_and_incomplete_descriptions_24k")
df_compinfo = ds_compinfo['train'].to_pandas()

df_features_with_year = pd.merge(
    df_features,
    df_compinfo[['__index_level_0__', 'year', 'cik', 'sic_code', 'ticker', 'company_name']],
    on='__index_level_0__',
    how='inner'
)
df_features_with_year['year'] = df_features_with_year['year'].astype(int)

train_f, val_f, test_f = split_by_year(df_features_with_year)
save_splits(train_f, val_f, test_f, "company_features", save_csv=False)

df_meta = df_compinfo[['__index_level_0__', 'cik', 'year', 'company_name', 'sic_code', 'ticker']].copy()
df_meta = df_meta.dropna(subset=['sic_code'])
df_meta['year'] = df_meta['year'].astype(int)

train_m, val_m, test_m = split_by_year(df_meta)
save_splits(train_m, val_m, test_m, "company_metadata")

log("Loading v1ctor10/cos_sim_4000pca_exp")
ds_pairs = load_dataset("v1ctor10/cos_sim_4000pca_exp")
df_pairs = ds_pairs['train'].to_pandas()
df_pairs = df_pairs.dropna(subset=['correlation'])
df_pairs['year'] = df_pairs['year'].astype(int)

train_p, val_p, test_p = split_by_year(df_pairs)
save_splits(train_p, val_p, test_p, "pairwise_similarities")

log("Done")
