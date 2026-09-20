# -*- coding: utf-8 -*-
"""
Pipeline: (1) Crop real-field backgrounds -> (2) Make SAM masks on lab images -> (3) Compose images

This is the Python version of 01_FBR_field_adaptive_background_recomposition.ipynb
Run it as: python 01_FBR_pipeline.py

Or use the .ipynb version in Jupyter.
"""

import os
import sys
import cv2
import pickle
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
SEED = 42
CROP_SIZE = 256
N_CROPS_PER_IMAGE = 1
SAM_CHECKPOINT = "sam_weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# Directories
LAB_DIR = Path("data/apple/PV")
LAB_IMG_DIR = LAB_DIR / "images"
COMPOSED_DIR = LAB_DIR / "bg_composed"
REAL_DIR = Path("data/apple/plantpathology")
REAL_IMG_DIR = REAL_DIR / "images"
BG_DIR = REAL_DIR / "cropped_bg"

# -----------------------------
# Utils
# -----------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(SEED); random.seed(SEED)

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# Background crops
# -----------------------------
def random_square_crops(
    in_dir,
    out_dir,
    crop_size=CROP_SIZE,
    n_crops_per_image=N_CROPS_PER_IMAGE,
    min_ratio=0.3,
    max_ratio=1.0,
    seed=SEED,
):
    from PIL import Image
    random.seed(seed)
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    files = [p for p in in_dir.iterdir() if p.suffix.lower() in exts]
    print(f"[info] background candidates: {len(files)} in {in_dir}")

    for idx, fp in enumerate(files, 1):
        try:
            with Image.open(fp) as img:
                try: img = ImageOps.exif_transpose(img)
                except: pass
                img = img.convert("RGB")
                w, h = img.size
                short = min(w, h)

                for k in range(n_crops_per_image):
                    if crop_size is None:
                        side = int(random.uniform(min_ratio, max_ratio) * short)
                        side = max(8, min(side, short))
                    else:
                        side = min(int(crop_size), short)

                    max_x, max_y = w - side, h - side
                    x0 = 0 if max_x <= 0 else random.randint(0, max_x)
                    y0 = 0 if max_y <= 0 else random.randint(0, max_y)

                    crop = img.crop((x0, y0, x0 + side, y0 + side))
                    out_name = f"{fp.stem}_crop{k:02d}_s{side}_x{x0}_y{y0}.jpg"
                    crop.save(out_dir / out_name, quality=95)
        except Exception as e:
            print(f"[warn] crop skip {fp.name}: {e}")

        if idx % 50 == 0:
            print(f"[info] cropped {idx}/{len(files)}")
            
    print("[done] background crops ->", out_dir)

# -----------------------------
# Mask postprocess
# -----------------------------
def adjust_mask(best_mask, dilate=15, erode=12, thresh=127):
    """best_mask: bool or 0/255"""
    m = best_mask
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    else:
        if m.max() not in (0, 1, 255):
            _, m = cv2.threshold(m, thresh, 255, cv2.THRESH_BINARY)
        else:
            m = (m > 0).astype(np.uint8) * 255

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
    m = cv2.dilate(m, k1)
    m = cv2.erode(m, k2)
    return m

# -----------------------------
# Composition
# -----------------------------
def image_composition(file_name, mask, lab_dir, bg_dir, out_size=256, blur_sigma=3):
    """Return RGB uint8"""
    bg_dir = Path(bg_dir)
    bg_files = [f for f in bg_dir.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}]
    if not bg_files:
        raise RuntimeError(f"No background images in {bg_dir}")

    bg_path = str(random.choice(bg_files))
    bg_bgr = cv2.imread(bg_path); assert bg_bgr is not None, f"read fail: {bg_path}"
    bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)
    if blur_sigma and blur_sigma > 0:
        bg = cv2.GaussianBlur(bg, (0,0), blur_sigma)

    lab_path = os.path.join(str(lab_dir), file_name)
    img_bgr = cv2.imread(lab_path); assert img_bgr is not None, f"read fail: {lab_path}"
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_r = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    bg_r  = cv2.resize(bg,  (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    m_r = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    m_bin = (m_r > 0).astype(np.uint8)

    out = img_r * m_bin[..., None] + bg_r * (1 - m_bin[..., None])
    return out.astype(np.uint8)

# -----------------------------
# Main
# -----------------------------
def main():
    # 0) dirs
    ensure_dirs(LAB_DIR, LAB_IMG_DIR, COMPOSED_DIR, REAL_DIR, REAL_IMG_DIR, BG_DIR)

    # 1) load SAM
    import torch
    sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device: {device}")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 2) crop backgrounds from real-field set
    random_square_crops(
        in_dir=REAL_IMG_DIR,
        out_dir=BG_DIR,
        crop_size=CROP_SIZE,
        n_crops_per_image=N_CROPS_PER_IMAGE,
        seed=SEED
    )

    # 3) generate masks with SAM (lab images)
    best_masks = {}
    file_names = [f for f in os.listdir(LAB_IMG_DIR) if not f.startswith(".")]

    print(f"[info] lab images: {len(file_names)} in {LAB_IMG_DIR}")
    for fname in tqdm(file_names, desc="SAM masks"):
        img_bgr = cv2.imread(str(LAB_IMG_DIR / fname))
        if img_bgr is None:
            print(f"[warn] skip (read fail): {fname}")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        # NOTE: SAM point_coords expects (x,y) = (col,row)
        input_point = np.array([
            [W/2, H/2],             # positive center
            [W/32, H/32],           # negatives: four corners
            [W/32, 31*H/32],
            [31*W/32, H/32],
            [31*W/32, 31*H/32],
        ], dtype=np.float32)
        input_label = np.array([1, 0, 0, 0, 0], dtype=np.int32)

        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_mask = masks[np.argmax(scores)]     # bool HxW
        best_masks[fname] = adjust_mask(best_mask)

        # incremental save (robust to interruption)
        save_pickle(best_masks, LAB_DIR / "pv_masks.pickle")

    print(f"[info] masks stored: {len(best_masks)}")

    # 4) compose
    best_masks = load_pickle(LAB_DIR / "pv_masks.pickle")

    bg_list = [f for f in os.listdir(BG_DIR) if not f.startswith(".")]
    if not bg_list:
        raise RuntimeError(f"[error] No backgrounds in {BG_DIR}")

    for fname in tqdm(file_names, desc="Compose"):
        if fname not in best_masks:
            print(f"[warn] mask missing: {fname} -> skip")
            continue

        comp = image_composition(fname, best_masks[fname], LAB_IMG_DIR, BG_DIR, out_size=CROP_SIZE, blur_sigma=3)
        out_path = str(COMPOSED_DIR / fname)
        ok = cv2.imwrite(out_path, cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        if not ok:
            print(f"[warn] save fail: {out_path}")

    print(f"[done] composed -> {COMPOSED_DIR}")

# -----------------------------
if __name__ == "__main__":
    main()
