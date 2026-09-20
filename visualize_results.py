"""
FBR-UDA: Visualize Training Results
====================================
Generates training curves, comparison charts, and confusion matrices.
Run AFTER training is complete (exp/ folder must exist).

Usage:
    python visualize_results.py

Output:
    results/
        training_curves/
            {model}_curves.png       — per-model loss/acc/F1 over epochs
        comparisons/
            all_models_comparison.png — side-by-side bar chart
            model_summary.csv        — summary table
        confusion_matrices/
            {model}_confusion.png    — per-model confusion matrix
        all_results.png             — single combined image
"""

import os
import csv
import glob
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from datasets import get_dataset
from models import get_model
from utils.transforms_utils import transform, augmentation

# ─── Config ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
EXP_DIR = PROJECT_ROOT / "exp"
RESULTS_DIR = PROJECT_ROOT / "results"
CROP = "apple"
N_CLASS = 3
CLASS_NAMES = ["healthy", "rust", "scab"]
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else
                       "cuda:0" if torch.cuda.is_available() else "cpu")

MODELS = ["ddc", "dcoral", "dann", "cdan", "daln"]
# DALN runs with different lambda_nwd values — find them from folder names
MODEL_DIRS = {}

def find_experiment_dirs():
    """Auto-discover all experiment directories and their CSV logs."""
    found = {}
    for entry in EXP_DIR.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name  # e.g. "apple_ddc", "apple_daln"
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        model = parts[1]
        # Find CSV files
        csvs = list(entry.glob("*.csv"))
        # Find best checkpoint
        best_ckpts = list(entry.glob("*_best.pth"))
        if csvs:
            found[name] = {
                "model": model,
                "dir": entry,
                "csv": csvs[0],
                "best_ckpt": best_ckpts[0] if best_ckpts else None,
            }
    return found


def load_csv_data(csv_path):
    """Load training CSV into dict of lists."""
    data = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(v)
    return dict(data)


# ─── 1. Per-Model Training Curves ─────────────────────
def plot_training_curves(exp_info, save_dir):
    """Plot train_loss, val_loss, val_acc, val_f1 for one model."""
    data = load_csv_data(exp_info["csv"])
    model_name = exp_info["model"]
    label = exp_info["dir"].name

    epochs = data.get("epoch", list(range(1, len(data.get("train_loss", [])) + 1)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves — {label.upper()}", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, data["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    ax.plot(epochs, data["val_loss"], label="Val Loss", color="#F44336", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, data["val_acc"], label="Val Accuracy", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1
    ax = axes[2]
    ax.plot(epochs, data["train_f1"], label="Train F1", color="#FF9800", linewidth=2, linestyle="--")
    ax.plot(epochs, data["val_f1"], label="Val F1", color="#9C27B0", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / f"{label}_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")
    return data


# ─── 2. Comparison Charts ─────────────────────────────
def plot_comparison(all_data, save_dir):
    """Bar chart comparing all models: best val_acc, best val_f1, best val_loss."""
    labels = []
    best_accs = []
    best_f1s = []
    best_losses = []
    final_accs = []
    final_f1s = []

    for name, data in sorted(all_data.items()):
        labels.append(name.upper())
        best_accs.append(max(data["val_acc"]))
        best_f1s.append(max(data["val_f1"]))
        best_losses.append(min(data["val_loss"]))
        final_accs.append(data["val_acc"][-1])
        final_f1s.append(data["val_f1"][-1])

    x = np.arange(len(labels))
    width = 0.35

    # Accuracy comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Model Comparison (All UDA Methods)", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    # Best Accuracy
    ax = axes[0]
    bars = ax.bar(x, best_accs, width=0.6, color=colors[:len(labels)])
    ax.set_ylabel("Accuracy")
    ax.set_title("Best Validation Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Best F1
    ax = axes[1]
    bars = ax.bar(x, best_f1s, width=0.6, color=colors[:len(labels)])
    ax.set_ylabel("F1 Score")
    ax.set_title("Best Validation F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, best_f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Best Loss (lower is better)
    ax = axes[2]
    bars = ax.bar(x, best_losses, width=0.6, color=colors[:len(labels)])
    ax.set_ylabel("Loss")
    ax.set_title("Best Validation Loss (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for bar, val in zip(bars, best_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = save_dir / "all_models_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")

    # Also save a combined training curves overlay
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Training Curves — All Models", fontsize=14, fontweight="bold")

    for i, (name, data) in enumerate(sorted(all_data.items())):
        c = colors[i % len(colors)]
        epochs = data.get("epoch", list(range(1, len(data["train_loss"]) + 1)))
        axes[0].plot(epochs, data["val_loss"], label=name.upper(), color=c, linewidth=2)
        axes[1].plot(epochs, data["val_acc"], label=name.upper(), color=c, linewidth=2)
        axes[2].plot(epochs, data["val_f1"], label=name.upper(), color=c, linewidth=2)

    axes[0].set_title("Validation Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].set_title("Validation F1"); axes[2].set_xlabel("Epoch"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / "all_models_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")

    # Save summary CSV
    csv_path = save_dir / "model_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Best_Val_Acc", "Best_Val_F1", "Best_Val_Loss", "Final_Val_Acc", "Final_Val_F1", "Epochs"])
        for i, (name, data) in enumerate(sorted(all_data.items())):
            w.writerow([name.upper(), f"{best_accs[i]:.4f}", f"{best_f1s[i]:.4f}",
                        f"{best_losses[i]:.4f}", f"{final_accs[i]:.4f}", f"{final_f1s[i]:.4f}",
                        len(data["epoch"])])
    print(f"  [saved] {csv_path}")

    return labels, best_accs, best_f1s


# ─── 3. Confusion Matrices ────────────────────────────
def plot_confusion_matrices(all_exp_info, save_dir):
    """Generate confusion matrices using best checkpoints."""
    from utils.train_config import transform as train_transform

    # Load test dataset once
    test_dataset = get_dataset(CROP, {"type": "tst", "transform": transform})
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    for name, info in sorted(all_exp_info.items()):
        if info["best_ckpt"] is None:
            print(f"  [skip] {name} — no checkpoint found")
            continue

        model = get_model(info["model"], "resnet18", N_CLASS).to(DEVICE)
        try:
            model.load_state_dict(torch.load(info["best_ckpt"], map_location=DEVICE))
        except Exception as e:
            print(f"  [skip] {name} — checkpoint load failed: {e}")
            continue
        model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(N_CLASS)))
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Confusion Matrix — {name.upper()}", fontsize=13, fontweight="bold")

        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
        axes[0].set_title("Counts")
        axes[0].set_ylabel("True")
        axes[0].set_xlabel("Predicted")

        # Normalized
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1],
                    vmin=0, vmax=1)
        axes[1].set_title("Normalized")
        axes[1].set_ylabel("True")
        axes[1].set_xlabel("Predicted")

        plt.tight_layout()
        out_path = save_dir / f"{name}_confusion.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {out_path}")

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        print(f"  [{name}] Per-class accuracy:")
        for cls_name, acc in zip(CLASS_NAMES, per_class_acc):
            print(f"    {cls_name}: {acc:.4f}")

    return


# ─── 4. Combined Summary Image ────────────────────────
def plot_combined_summary(all_data, all_exp_info, save_dir):
    """Single large image with everything."""
    n_models = len(all_data)
    fig = plt.figure(figsize=(24, 6 * (n_models + 1)))
    gs = gridspec.GridSpec(n_models + 1, 3, hspace=0.4, wspace=0.3)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

    # Top row: comparison bars
    labels = []
    best_accs = []
    best_f1s = []
    for name, data in sorted(all_data.items()):
        labels.append(name.upper())
        best_accs.append(max(data["val_acc"]))
        best_f1s.append(max(data["val_f1"]))

    x = np.arange(len(labels))

    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(x, best_accs, color=colors[:len(labels)], width=0.6)
    ax.set_title("Best Val Accuracy", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(x, best_f1s, color=colors[:len(labels)], width=0.6)
    ax.set_title("Best Val F1", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, best_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Overlay curves
    ax = fig.add_subplot(gs[0, 2])
    for i, (name, data) in enumerate(sorted(all_data.items())):
        epochs = data.get("epoch", list(range(1, len(data["val_acc"]) + 1)))
        ax.plot(epochs, data["val_acc"], label=name.upper(), color=colors[i % len(colors)], linewidth=2)
    ax.set_title("Val Accuracy Overlay", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Per-model rows
    for row_idx, (name, data) in enumerate(sorted(all_data.items()), start=1):
        epochs = data.get("epoch", list(range(1, len(data["train_loss"]) + 1)))

        ax = fig.add_subplot(gs[row_idx, 0])
        ax.plot(epochs, data["train_loss"], label="Train", color="#2196F3", linewidth=2)
        ax.plot(epochs, data["val_loss"], label="Val", color="#F44336", linewidth=2)
        ax.set_title(f"{name.upper()} — Loss", fontweight="bold", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[row_idx, 1])
        ax.plot(epochs, data["val_acc"], color="#4CAF50", linewidth=2)
        ax.set_title(f"{name.upper()} — Accuracy", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[row_idx, 2])
        ax.plot(epochs, data["val_f1"], color="#9C27B0", linewidth=2)
        ax.set_title(f"{name.upper()} — F1", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle("FBR-UDA — Complete Training Results", fontsize=16, fontweight="bold", y=1.01)
    out_path = save_dir / "all_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ─── Main ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("FBR-UDA: Visualizing Training Results")
    print("=" * 60)

    # Discover experiments
    exp_dirs = find_experiment_dirs()
    if not exp_dirs:
        print(f"[error] No experiment directories found in {EXP_DIR}")
        print("  Make sure you have trained models in exp/")
        return

    print(f"\n[found] {len(exp_dirs)} experiments:")
    for name, info in exp_dirs.items():
        ckpt_status = "yes" if info["best_ckpt"] else "no"
        print(f"  {name}: model={info['model']}, checkpoint={ckpt_status}")

    # Create output dirs
    curves_dir = RESULTS_DIR / "training_curves"
    comp_dir = RESULTS_DIR / "comparisons"
    cm_dir = RESULTS_DIR / "confusion_matrices"
    for d in [curves_dir, comp_dir, cm_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Per-model training curves
    print("\n--- Training Curves ---")
    all_data = {}
    for name, info in exp_dirs.items():
        data = plot_training_curves(info, curves_dir)
        all_data[name] = data

    # 2. Comparison charts
    print("\n--- Model Comparisons ---")
    plot_comparison(all_data, comp_dir)

    # 3. Confusion matrices
    print("\n--- Confusion Matrices ---")
    plot_confusion_matrices(exp_dirs, cm_dir)

    # 4. Combined summary
    print("\n--- Combined Summary ---")
    plot_combined_summary(all_data, exp_dirs, RESULTS_DIR)

    print(f"\n{'=' * 60}")
    print(f"All results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
