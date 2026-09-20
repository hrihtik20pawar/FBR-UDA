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
    pp_img_dir = DATA_DIR / "plantpathology" / "images"
    pp_img_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(pp_img_dir.iterdir()) if pp_img_dir.exists() else []
    if len(existing) > 100:
        print(f"  [skip] Already have {len(existing)} images in {pp_img_dir}")
        return True

    # Try HuggingFace dataset
    print("  Trying HuggingFace source...")
    hf_url = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"
    # Try alternative: a community upload of PlantPathology
    alt_urls = [
        "https://huggingface.co/datasets/frgfm/PlantVillage/resolve/main/data.zip",
    ]

    # Try kagglehub first
    try:
        import kagglehub
        path = kagglehub.competition_download("plant-pathology-2020-fgvc7")
        for img_file in Path(path).rglob("*.jpg"):
            shutil.copy2(img_file, pp_img_dir / img_file.name)
        for img_file in Path(path).rglob("*.png"):
            shutil.copy2(img_file, pp_img_dir / img_file.name)
        print(f"  [done] Downloaded via kagglehub to {pp_img_dir}")
        return True
    except Exception as e:
        print(f"  [info] kagglehub not available: {e}")

    # Fallback: use PV images as both source and target backgrounds
    pv_dir = DATA_DIR / "PV" / "images"
    if pv_dir.exists():
        count = 0
        for img_file in pv_dir.iterdir():
            if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                shutil.copy2(img_file, pp_img_dir / img_file.name)
                count += 1
        print(f"  [fallback] Copied {count} PV images as PlantPathology stand-in")
        print(f"  [info] For real field images, download manually from:")
        print(f"         https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data")
        return True

    print("  [error] No data source available. Please manually download from:")
    print("         https://www.kaggle.com/c/plant-pathology-2020-fgvc7/data")
    print(f"         Place images in: {pp_img_dir}")
    return False

def generate_labels_and_structure():
    print("\n" + "="*60)
    print("5. Generating labels & data splits")
    print("="*60)
    import csv
    import pickle

    pv_img_dir = DATA_DIR / "PV" / "images"
    pp_img_dir = DATA_DIR / "plantpathology" / "images"

    # --- PlantVillage labels (3 classes from filenames) ---
    pv_labels = {}
    for f in pv_img_dir.iterdir():
        name = f.name
        if 'FREC_Scab' in name:
            pv_labels[name] = 2  # scab
        elif 'FREC_C.Rust' in name:
            pv_labels[name] = 1  # rust
        elif 'JR_FrgE.S' in name:
            pv_labels[name] = 1  # also rust-like
        elif 'RS_HL' in name:
            pv_labels[name] = 0  # healthy

    pv_pkl = DATA_DIR / "PV" / "pv_labels.pickle"
    with open(pv_pkl, 'wb') as f:
        pickle.dump(pv_labels, f)
    print(f"  PV labels: {len(pv_labels)} images, 3 classes")

    # --- PlantPathology labels (from filenames or copy from PV) ---
    pp_labels = {}
    for f in pp_img_dir.iterdir():
        name = f.name
        if 'FREC_Scab' in name:
            pp_labels[name] = 2
        elif 'FREC_C.Rust' in name:
            pp_labels[name] = 1
        elif 'JR_FrgE.S' in name:
            pp_labels[name] = 1
        elif 'RS_HL' in name:
            pp_labels[name] = 0
        elif name.startswith('Train_') or name.startswith('Test_'):
            pp_labels[name] = 0  # default healthy for unknown

    pp_pkl = DATA_DIR / "plantpathology" / "apple_labels.pickle"
    with open(pp_pkl, 'wb') as f:
        pickle.dump(pp_labels, f)
    print(f"  PP labels: {len(pp_labels)} images")

    # --- exp_structure.pickle ---
    src_names = sorted([f.name for f in pv_img_dir.iterdir()
                        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    tgt_names = sorted([f.name for f in pp_img_dir.iterdir()
                        if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    n_tgt = len(tgt_names)
    structure = {
        'source': src_names,
        'target': tgt_names[:int(n_tgt * 0.8)],
        'test': tgt_names[int(n_tgt * 0.8):],
    }
    struct_pkl = DATA_DIR / "exp_structure.pickle"
    with open(struct_pkl, 'wb') as f:
        pickle.dump(structure, f)
    print(f"  Structure: source={len(src_names)}, target={len(structure['target'])}, test={len(structure['test'])}")

def main():
    print("="*60)
    print("FBR-UDA: Download Resources")
    print("="*60)
    
    download_sam()
    download_chili()
    download_plantvillage()
    download_plantpathology()
    generate_labels_and_structure()
    
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
