import os
import urllib.request
import zipfile

DATASET_URL = "https://github.com/scikit-image/scikit-image-data/raw/main/multiview_cup.zip"
TARGET_DIR = "./sample_drone_data"

def download_and_extract():
    os.makedirs(TARGET_DIR, exist_ok=True)
    zip_path = "sample_data.zip"
    
    print("[+] Downloading calibrated multi-view test images...")
    # Small calibrated test dataset (~5MB)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/colmap/colmap/master/src/colmap/test_data/datasets/south-building.zip", 
        zip_path
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    
    os.remove(zip_path)
    print(f"[✓] Extracted test dataset to: '{os.path.abspath(TARGET_DIR)}'")

if __name__ == "__main__":
    download_and_extract()