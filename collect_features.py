"""
This file pulls the Llama features from the HuggingFace dataset repository that was used by the parent paper. It outputs to my Google Drive. The Llama features, however, are not being referenced for the majority of the other files, since it is too big to remain in Drive, so this features file was also moved to the HuggingFace.
"""

from datasets import load_dataset
ds = load_dataset("marco-molinari/company_reports_with_features")
df = ds["train"].to_pandas()[["__index_level_0__", "features"]] # Specifically, this line pulls out all of the features for each report.
df.to_pickle("/content/drive/MyDrive/company_similarity_sae/data/llama_features.pkl")