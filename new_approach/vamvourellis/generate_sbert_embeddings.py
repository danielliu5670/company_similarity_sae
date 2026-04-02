#!/usr/bin/env python3
"""
Generate SBERT embeddings (Vamvourellis et al. baseline).

Usage:
    !pip install sentence-transformers
    !python generate_sbert_embeddings.py \
        --text-col description \
        --out-pkl /content/drive/MyDrive/company_similarity_sae/data/sbert_embeddings.pkl
"""

import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument("--out-pkl", required=True)
P.add_argument("--text-col", required=True)
P.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
P.add_argument("--batch-size", type=int, default=64)
P.add_argument("--features-ds",
               default="marco-molinari/company_reports_with_features")
args = P.parse_args()

ds = load_dataset(args.features_ds)["train"]
if args.text_col not in ds.column_names:
    raise ValueError(
        f"Column '{args.text_col}' not found. Available: {ds.column_names}")

df = ds.to_pandas()
df["__index_level_0__"] = df["__index_level_0__"].astype(str)
texts = df[args.text_col].fillna("").astype(str).tolist()
ids = df["__index_level_0__"].tolist()

model = SentenceTransformer(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
max_len = model.max_seq_length
embed_dim = model.get_sentence_embedding_dimension()

all_chunks = []
chunk_to_text = []
chunk_size = max_len - 2

for i, text in enumerate(texts):
    if not text.strip():
        all_chunks.append("")
        chunk_to_text.append(i)
        continue
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= chunk_size:
        all_chunks.append(text)
        chunk_to_text.append(i)
    else:
        for j in range(0, len(tokens), chunk_size):
            ct = tokens[j:j + chunk_size]
            all_chunks.append(
                tokenizer.decode(ct, skip_special_tokens=True))
            chunk_to_text.append(i)

chunk_embeddings = model.encode(
    all_chunks, batch_size=args.batch_size,
    show_progress_bar=True, normalize_embeddings=False,
)

chunk_to_text = np.array(chunk_to_text)
embeddings = np.zeros((len(texts), embed_dim), dtype=np.float64)
counts = np.zeros(len(texts), dtype=np.int32)
for ci in range(len(all_chunks)):
    embeddings[chunk_to_text[ci]] += chunk_embeddings[ci]
    counts[chunk_to_text[ci]] += 1
counts = np.clip(counts, 1, None)
embeddings /= counts[:, np.newaxis]
embeddings = embeddings.astype(np.float32)

out_df = pd.DataFrame({
    "__index_level_0__": ids,
    "sbert_embedding": list(embeddings),
})
out_df.to_pickle(args.out_pkl)