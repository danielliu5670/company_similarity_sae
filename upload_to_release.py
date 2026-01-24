#!/usr/bin/env python3
import os
import subprocess
import sys
import glob

REPO = "danielliu5670/company_similarity_sae"
TAG = "data-v1"
SPLIT_DIR = "data/input/split"
FOLDERS = ["train", "test", "val"]

def main():
    subprocess.run(["gh", "release", "view", TAG, "--repo", REPO], capture_output=True)
    result = subprocess.run(["gh", "release", "view", TAG, "--repo", REPO], capture_output=True)
    
    if result.returncode != 0:
        subprocess.run([
            "gh", "release", "create", TAG,
            "--repo", REPO,
            "--title", "Dataset Files",
            "--notes", "Split data files for company_similarity_sae"
        ], check=True)
    
    for folder in FOLDERS:
        folder_path = os.path.join(SPLIT_DIR, folder)
        files = sorted(glob.glob(os.path.join(folder_path, "*.pkl.*")))
        
        for f in files:
            print(os.path.basename(f))
            subprocess.run([
                "gh", "release", "upload", TAG, f,
                "--repo", REPO,
                "--clobber"
            ], check=True)

if __name__ == "__main__":
    main()
