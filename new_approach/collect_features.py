from datasets import load_dataset
ds = load_dataset("marco-molinari/company_reports_with_features")
df = ds["train"].to_pandas()[["__index_level_0__", "features"]]
df.to_pickle("/content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl")