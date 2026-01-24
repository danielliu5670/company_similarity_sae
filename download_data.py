#!/usr/bin/env python3
import os
import subprocess
import sys

REPO = "danielliu5670/company_similarity_sae"
TAG = "data-v1"
FOLDERS = ["train", "test", "val"]
OUTPUT_DIR = "data/input/split"

def main():
    for folder in FOLDERS:
        out_folder = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(out_folder, exist_ok=True)
    
    result = subprocess.run(
        ["gh", "release", "view", TAG, "--repo", REPO, "--json", "assets", "-q", ".assets[].name"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Failed to fetch release: {result.stderr}")
        sys.exit(1)
    
    assets = result.stdout.strip().split("\n")
    
    for asset in assets:
        if not asset:
            continue
        for folder in FOLDERS:
            if f"_{folder}.pkl." in asset:
                dest = os.path.join(OUTPUT_DIR, folder, asset)
                break
        else:
            continue
        
        if os.path.exists(dest):
            continue
        
        print(asset)
        subprocess.run(
            ["gh", "release", "download", TAG, "--repo", REPO, "--pattern", asset, "--dir", os.path.dirname(dest)],
            check=True
        )

if __name__ == "__main__":
    main()
