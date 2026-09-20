# FBR-UDA — Field-Adaptive Background Recomposition + Unsupervised Domain Adaptation

> **Bridging the Lab-to-Field gap in plant disease diagnosis through unsupervised domain adaptation enhanced by background recomposition**
>
> Paper: https://doi.org/10.1016/j.ecoinf.2025.103579

---

## Overview

Laboratory images (e.g., **PlantVillage**) often fail in real fields (e.g., **PlantPathology**) due to background/lighting/occlusion gaps. This repo provides two complementary components:

1. **FBR — Field-Adaptive Background Recomposition**: Segment foreground leaves with **SAM**, crop real-field backgrounds, and composite them to create more field-like training samples.
2. **UDA — Unsupervised Domain Adaptation**: Train with source labeled (lab) and target unlabeled (field) data using methods: **DDC**, **DCORAL**, **DANN**, **CDAN**, and **DALN**.

---

## Quick Start

### 1) Setup environment

```bash
cd FBR-UDA
python setup.py
# Or manually:
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2) Download all resources (datasets + SAM weights)

```bash
python download_all.py
```

This downloads:
- **SAM ViT-H weights** (~2.4 GB) → `sam_weights/`
- **PlantVillage** (Apple subset) → `data/apple/PV/images/`
- **PlantPathology** (field images) → `data/apple/plantpathology/images/`
- **P-Chili Pepper dataset** from Zenodo

If auto-download fails, manually download from:
- PlantVillage: https://www.kaggle.com/datasets/emmarex/plantdisease
- PlantPathology: https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7
- SAM weights: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

### 3) Generate FBR data

```bash
python 01_FBR_pipeline.py
```

This will:
1. Crop real-field backgrounds from target images
2. Segment leaf foregrounds from lab images using SAM
3. Composite foregrounds onto backgrounds → `data/apple/PV/bg_composed/`

### 4) Train & evaluate

```bash
python main.py
```

Artifacts (checkpoints, logs) are saved under `exp/`.

---

## Data Layout

```
data/
└── apple/
    ├── PV/                         # lab-collected (PlantVillage)
    │   ├── images/                 # raw lab images
    │   ├── bg_composed/            # FBR outputs (composited images)
    │   ├── pv_masks.pickle         # SAM masks index
    │   └── pv_labels.pickle        # labels for PV
    └── plantpathology/             # real-field (PlantPathology)
        ├── images/                 # raw field images
        ├── cropped_bg/             # background crops used for FBR
        └── apple_labels.pickle     # labels for field set
```

---

## Supported Models

| Model | Description |
|-------|-------------|
| `ddc` | Deep Domain Confusion (MMD) |
| `dcoral` | Deep CORAL |
| `dann` | Domain-Adversarial Neural Network |
| `cdan` | Conditional Adversarial Domain Adaptation |
| `daln` | Domain-Adaptive Learning Network (NWD) |
| `vanilaresnet` | Baseline ResNet (no adaptation) |

---

## Configuration

Default options in `utils/train_config.py`. Key settings:

```python
args = dict(
    crop='apple',
    src_dataset={'kwargs': {'type': 'src_bg_augmented'}},  # FBR-composed
    tgt_dataset={'kwargs': {'type': 'tgt'}},               # field images
    test_dataset={'kwargs': {'type': 'tst'}},              # held-out field
    model={'name': 'dann', 'backbone': 'resnet18', 'n_class': 3},
    n_epochs=300,
    early_stop=25,
)
```

---

## Citation

```bibtex
@article{JEON2026103579,
   title = {Bridging the Lab-to-Field gap in plant disease diagnosis through unsupervised domain adaptation enhanced by background recomposition},
   journal = {Ecological Informatics},
   volume = {93},
   pages = {103579},
   year = {2026},
   doi = {https://doi.org/10.1016/j.ecoinf.2025.103579},
}
```
