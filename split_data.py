import os
import glob

CHUNK_SIZE = 90 * 1024 * 1024
INPUT_DIR = "data/input"
SPLIT_DIR = "data/input/split"
FOLDERS = ["train", "test", "val"]

def split_file(filepath, output_dir):
    filename = os.path.basename(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(filepath, 'rb') as f:
        chunk_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            chunk_path = os.path.join(output_dir, f"{filename}.{chunk_num:03d}")
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk)
            chunk_num += 1
    
    os.remove(filepath)
    print(f"{filename}: {chunk_num} chunks")

def main():
    for folder in FOLDERS:
        folder_path = os.path.join(INPUT_DIR, folder)
        output_folder = os.path.join(SPLIT_DIR, folder)
        
        for filepath in glob.glob(os.path.join(folder_path, "*")):
            if os.path.isfile(filepath):
                split_file(filepath, output_folder)

if __name__ == "__main__":
    main()
