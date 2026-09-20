"""
FBR-UDA: Complete Setup & Run Script
=====================================
Sets up environment, downloads data, runs FBR pipeline, and trains all UDA models.

Usage (run from project root):
    python run_all.py

Or step-by-step:
    python run_all.py --step setup
    python run_all.py --step download
    python run_all.py --step fbr
    python run_all.py --step train
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
VENV_DIR = PROJECT_ROOT / "venv"

# ─────────────────────────────────────────────────────
# Step 1: Environment Setup
# ─────────────────────────────────────────────────────
def get_python():
    """Get the correct python executable (venv or system)."""
    if platform.system() == "Windows":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def setup_environment():
    print("=" * 60)
    print("STEP 1: Setting up virtual environment")
    print("=" * 60)

    if not VENV_DIR.exists():
        print("[1/4] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    else:
        print("[1/4] Virtual environment already exists.")

    python = get_python()

    print("[2/4] Upgrading pip...")
    subprocess.run([python, "-m", "pip", "install", "--upgrade", "pip"], check=False)

    print("[3/4] Installing PyTorch with CUDA support...")
    subprocess.run([
        python, "-m", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=False)

    print("[4/4] Installing remaining dependencies...")
    deps = [
        "opencv-python", "numpy", "tqdm", "pillow",
        "torchmetrics", "segment-anything", "pytz", "requests"
    ]
    subprocess.run([python, "-m", "pip", "install"] + deps, check=False)

    print("\n[done] Environment ready.")
    print(f"  Python: {python}")
    return python


# ─────────────────────────────────────────────────────
# Step 2: Download Data
# ─────────────────────────────────────────────────────
def download_data(python):
    print("\n" + "=" * 60)
    print("STEP 2: Downloading datasets & weights")
    print("=" * 60)
    # Check if data already exists
    data_dir = PROJECT_ROOT / "data" / "apple" / "PV" / "images"
    sam_file = PROJECT_ROOT / "sam_weights" / "sam_vit_h_4b8939.pth"
    if data_dir.exists() and len(list(data_dir.iterdir())) > 100 and sam_file.exists():
        print("[skip] Data and weights already downloaded.")
        return
    subprocess.run([python, str(PROJECT_ROOT / "download_all.py")], check=False)


# ─────────────────────────────────────────────────────
# Step 3: FBR Pipeline
# ─────────────────────────────────────────────────────
def run_fbr_pipeline(python):
    print("\n" + "=" * 60)
    print("STEP 3: Running FBR Pipeline (SAM + Composition)")
    print("=" * 60)
    print("[info] This will take a long time on CPU (~hours for 4816 images).")
    subprocess.run([python, str(PROJECT_ROOT / "01_FBR_pipeline.py")], check=False)


# ─────────────────────────────────────────────────────
# Step 4: Train UDA Models
# ─────────────────────────────────────────────────────
def train_models(python):
    print("\n" + "=" * 60)
    print("STEP 4: Training UDA Models (DDC/DCORAL/DANN/CDAN/DALN)")
    print("=" * 60)
    subprocess.run([python, str(PROJECT_ROOT / "main.py")], check=False)


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FBR-UDA: Full Setup & Run")
    parser.add_argument("--step", choices=["setup", "download", "fbr", "train", "all"],
                        default="all", help="Which step to run (default: all)")
    args = parser.parse_args()

    python = get_python()

    if args.step in ("setup", "all"):
        python = setup_environment()

    if args.step in ("download", "all"):
        download_data(python)

    if args.step in ("fbr", "all"):
        run_fbr_pipeline(python)

    if args.step in ("train", "all"):
        train_models(python)

    if args.step == "all":
        print("\n" + "=" * 60)
        print("ALL DONE!")
        print("=" * 60)
        print("Results saved in: exp/apple_*/")


if __name__ == "__main__":
    main()
