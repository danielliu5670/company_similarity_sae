#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _get_param(params, *keys):
    for k in keys:
        if k in params:
            return params[k]
    raise KeyError(
        f"None of {keys} found in params. Available keys: {sorted(params.keys())}"
    )


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.W_enc = _get_param(params, "W_enc", "w_enc").to(device).float()
        self.b_enc = _get_param(params, "b_enc").to(device).float()
        self.W_dec = _get_param(params, "W_dec", "w_dec").to(device).float()
        self.b_dec = _get_param(params, "b_dec").to(device).float()
        raw_threshold = _get_param(params, "threshold", "log_threshold").to(device).float()
        if "log_threshold" in params and "threshold" not in params:
            self.threshold = torch.exp(raw_threshold)
        else:
            self.threshold = raw_threshold

    @torch.no_grad()
    def encode(self, x):
        x = x.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        pre_acts = torch.nan_to_num(pre_acts, nan=0.0, posinf=0.0, neginf=0.0)
        return pre_acts * (pre_acts > self.threshold)


def load_gemma_scope_sae(repo_id, sae_path, device="cpu"):
    npz_name = f"{sae_path}/params.npz"
    st_name = f"{sae_path}/params.safetensors"
    try:
        path = hf_hub_download(repo_id=repo_id, filename=st_name)
        params = load_file(path, device="cpu")
    except Exception:
        path = hf_hub_download(repo_id=repo_id, filename=npz_name)
        raw = dict(np.load(path))
        params = {k: torch.from_numpy(v) for k, v in raw.items()}
    return JumpReLUSAE(params, device=device)


P = argparse.ArgumentParser()
P.add_argument("--model", default="google/gemma-3-12b-pt")
P.add_argument("--sae-repo", default="google/gemma-scope-2-12b-pt")
P.add_argument("--sae-path", default="resid_post/layer_41_width_65k_l0_medium")
P.add_argument("--layer", type=int, default=41)
P.add_argument("--batch-size", type=int, default=2)
P.add_argument("--max-length", type=int, default=512)
P.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
P.add_argument("--output", default="data/gemma_features.pkl")
P.add_argument(
    "--descriptions-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
P.add_argument("--text-column", default="section_1")
args = P.parse_args()

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
device = torch.device(args.device)

print(f"Loading SAE: {args.sae_repo} / {args.sae_path}")
sae = load_gemma_scope_sae(args.sae_repo, args.sae_path, device)
sae_dim = sae.W_enc.shape[1]
print(f"  SAE feature dimension: {sae_dim}")

print(f"Loading model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map=args.device,
)
model.eval()

print("Loading company descriptions...")
ds = load_dataset(args.descriptions_ds)
df = ds["train"].to_pandas()

if args.text_column not in df.columns:
    text_cols = [c for c in df.columns if df[c].dtype == object]
    raise ValueError(
        f"Column '{args.text_column}' not found. Text-like columns: {text_cols}"
    )

features_list = []
index_list = []
nan_count = 0

with torch.no_grad():
    for start in tqdm(range(0, len(df), args.batch_size), desc="Extracting features"):
        batch = df.iloc[start : start + args.batch_size]
        texts = batch[args.text_column].fillna("").tolist()

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[args.layer + 1].float()
        attn_mask = inputs["attention_mask"]

        batch_sae = []
        for i in range(hidden.shape[0]):
            valid_hidden = hidden[i][attn_mask[i].bool()]
            if torch.isnan(valid_hidden).any() or torch.isinf(valid_hidden).any():
                valid_hidden = torch.nan_to_num(
                    valid_hidden, nan=0.0, posinf=0.0, neginf=0.0
                )
                nan_count += 1
            token_acts = sae.encode(valid_hidden)  # (num_valid_tokens, sae_dim)
            batch_sae.append(token_acts.sum(dim=0))  # sum across tokens

        sae_features = torch.stack(batch_sae)

        for i in range(len(batch)):
            idx = batch.iloc[i]["__index_level_0__"]
            features_list.append(sae_features[i].cpu().numpy())
            index_list.append(idx)

if nan_count > 0:
    print(f"  Warning: {nan_count} batches had NaN/Inf hidden states (replaced with 0)")

result_df = pd.DataFrame(
    {"__index_level_0__": index_list, "features": [[f] for f in features_list]}
)
result_df.to_pickle(args.output)
print(f"Saved {len(result_df)} feature vectors (dim={sae_dim}) to {args.output}")
