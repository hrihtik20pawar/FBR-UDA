import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
import requests
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "apple"
SAM_DIR = PROJECT_ROOT / "sam_weights"

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_CHECKPOINT = SAM_DIR / "sam_vit_h_4b8939.pth"

# Zenodo Chili dataset
CHILI_URL = "https://zenodo.org/records/17901489/files/Chili_Pepper_BBWV2_Flooding.zip?download=1"

def download_file(url, dest_path, desc="Downloading"):
    if dest_path.exists():
        print(f"  [skip] {dest_path.name} already exists ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [download] {desc}...")
    try:
        response = requests.get(url, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"  [done] Saved to {dest_path} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [error] Download failed: {e}")
        return False

def download_sam():
    print("\n" + "="*60)
    print("1. Downloading SAM ViT-H Weights (~2.4 GB)")
    print("="*60)
    SAM_DIR.mkdir(parents=True, exist_ok=True)
    return download_file(SAM_URL, SAM_CHECKPOINT, "SAM ViT-H checkpoint")

def download_chili():
    print("\n" + "="*60)
    print("2. Downloading P-Chili Pepper Dataset from Zenodo (~90 MB)")
    print("="*60)
    zip_path = PROJECT_ROOT / "Chili_Pepper_BBWV2_Flooding.zip"
    ok = download_file(CHILI_URL, zip_path, "Chili Pepper Dataset")
    if ok and zip_path.exists():
        print("  [extract] Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path=PROJECT_ROOT)
        print("  [done] Extracted to", PROJECT_ROOT)
        zip_path.unlink()
    return ok

def download_plantvillage():
    print("\n" + "="*60)
    print("3. Downloading PlantVillage Dataset (Apple subset)")
    print("="*60)
    print("  PlantVillage requires Kaggle API or manual download.")
    print("  Trying HuggingFace mirror...")
    
    hf_url = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
    zip_path = PROJECT_ROOT / "plantvillage_data.zip"
    ok = download_file(hf_url, zip_path, "PlantVillage from HuggingFace")
    
    if ok and zip_path.exists():
        print("  [extract] Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=PROJECT_ROOT / "plantvillage_raw")
            print("  [done] Extracted to plantvillage_raw/")
            zip_path.unlink()
            
            # Find and copy apple images
            pv_img_dir = DATA_DIR / "PV" / "images"
            pv_img_dir.mkdir(parents=True, exist_ok=True)
            
            raw_dir = PROJECT_ROOT / "plantvillage_raw"
            apple_dirs = list(raw_dir.rglob("Apple*"))
            count = 0
            for apple_dir in apple_dirs:
                if apple_dir.is_dir():
                    for img_file in apple_dir.iterdir():
                        if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                            shutil.copy2(img_file, pv_img_dir / img_file.name)
                            count += 1
            print(f"  [done] Copied {count} apple images to {pv_img_dir}")
        except Exception as e:
            print(f"  [error] Extraction failed: {e}")
            print("  [info] Manual download needed from:")
            print("         https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    else:
        print("  [info] Auto-download failed. Please manually download from:")
        print("         https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print(f"         Place Apple images in: {DATA_DIR / 'PV' / 'images'}")
    return ok

def download_plantpathology():
    print("\n" + "="*60)
    print("4. PlantPathology Dataset")
    print("="*60)
    print("  PlantPathology requires Kaggle API (competition data).")
    print("  Trying alternative source...")
    
    # Try kagglehub or direct
    try:
        import kagglehub
        path = kagglehub.competition_download("plant-pathology-2020-fgvc7")
        pp_img_dir = DATA_DIR / "plantpathology" / "images"
        pp_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for img_file in Path(path).rglob("*.jpg"):
            shutil.copy2(img_file, pp_img_dir / img_file.name)
        for img_file in Path(path).rglob("*.png"):
            shutil.copy2(img_file, pp_img_dir / img_file.name)
        print(f"  [done] Downloaded via kagglehub to {pp_img_dir}")
        return True
    except Exception as e:
        print(f"  [info] kagglehub not available or failed: {e}")
        print("  [info] Please manually download from:")
        print("         https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data")
        print(f"         Place images in: {DATA_DIR / 'plantpathology' / 'images'}")
        return False

def main():
    print("="*60)
    print("FBR-UDA: Download Resources")
    print("="*60)
    
    download_sam()
    download_chili()
    download_plantvillage()
    download_plantpathology()
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify data in data/apple/PV/images/")
    print("2. Verify data in data/apple/plantpathology/images/")
    print("3. Run: python 01_FBR_pipeline.py")
    print("4. Run: python main.py")

if __name__ == "__main__":
    main()
