import pandas as pd
import os
import urllib.request
from tqdm import tqdm

def main():
    metadata_path = "data/subset_multiclass_metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Metadata file {metadata_path} not found. Please run create_multiclass_dataset.py first.")
        return
        
    df = pd.read_csv(metadata_path)
    print(f"Downloading {len(df)} records (100Hz)...")
    
    base_url = "https://physionet.org/files/ptb-xl/1.0.3/"
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    
    # We will only download what we don't already have.
    downloaded = 0
    failed = 0
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename_lr = row['filename_lr']
        file_path = os.path.dirname(filename_lr)
        os.makedirs(os.path.join(out_dir, file_path), exist_ok=True)
        
        for ext in ['.dat', '.hea']:
            url = f"{base_url}{filename_lr}{ext}"
            dest = os.path.join(out_dir, f"{filename_lr}{ext}")
            
            if not os.path.exists(dest):
                try:
                    urllib.request.urlretrieve(url, dest)
                    downloaded += 1
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
                    failed += 1
                    
    print(f"Finished downloading. Downloaded: {downloaded} files, Failed: {failed}")

if __name__ == '__main__':
    main()
