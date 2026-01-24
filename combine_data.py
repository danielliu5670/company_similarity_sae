import os
import glob
from collections import defaultdict

SPLIT_DIR = "data/input/split"
OUTPUT_DIR = "data/input"
FOLDERS = ["train", "test", "val"]

def combine_chunks(chunk_paths, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunk_paths = sorted(chunk_paths)
    
    with open(output_path, 'wb') as out_file:
        for chunk_path in chunk_paths:
            with open(chunk_path, 'rb') as chunk_file:
                out_file.write(chunk_file.read())

def main():
    for folder in FOLDERS:
        split_folder = os.path.join(SPLIT_DIR, folder)
        output_folder = os.path.join(OUTPUT_DIR, folder)
        
        if not os.path.exists(split_folder):
            continue
        
        files = defaultdict(list)
        for chunk_path in glob.glob(os.path.join(split_folder, "*")):
            base_name = ".".join(os.path.basename(chunk_path).rsplit(".", 1)[:-1])
            files[base_name].append(chunk_path)
        
        for filename, chunks in files.items():
            output_path = os.path.join(output_folder, filename)
            combine_chunks(chunks, output_path)
            print(f"{filename}")

if __name__ == "__main__":
    main()
