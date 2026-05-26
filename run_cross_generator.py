"""Chạy thực nghiệm Cross-Generator và lưu kết quả vào tests/cross_generator/."""
import os
import sys
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

# --- Reproducibility ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"torch: {torch.__version__}")
print(f"timm: {timm.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

CHECKPOINT_PATH = "artifacts/checkpoints/best_stage3.pth"
TEST_CSV = "data/splits/test.csv"
BATCH_SIZE = 64
OUTPUT_DIR = "reports/cross_generator"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_KEYS = {
    "140k_stylegan": "140k-real-and-fake-faces",
    "deepfake_real": "deepfake-and-real-images",
    "hard_fakereal": "hardfakevsrealfaces",
    "ciplab": "real-and-fake-face-detection",
}

# --- Model ---
sys.path.insert(0, "src")
from model import FaceDetectionModel


def load_model(checkpoint_path, device):
    model = FaceDetectionModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        hidden_dim=256,
        dropout=0.5,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Loaded checkpoint — epoch {ckpt['epoch']}, "
        f"val_loss {ckpt['val_loss']:.4f}, val_acc {ckpt['val_acc']:.4f}"
    )
    model.eval()
    return model


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


# --- Dataset ---
class FaceDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {"Real": 0, "Fake": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = np.array(Image.open(row["image_path"]).convert("RGB"))
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img_t = self.transform(image=img)["image"]
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.float32)
        return img_t, label


# --- Evaluate ---
def evaluate(model, loader, device):
    all_preds, all_probs, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            if (batch_idx + 1) % 20 == 0:
                print(f"    batch {batch_idx + 1}/{len(loader)}", flush=True)
    y = np.array(all_labels)
    p = np.array(all_preds)
    prob = np.array(all_probs)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)),
        "y_true": y,
        "y_pred": p,
        "y_prob": prob,
    }


# --- Main ---
model = load_model(CHECKPOINT_PATH, DEVICE)
test_df = pd.read_csv(TEST_CSV)


def assign_dataset_key(path):
    for key, prefix in DATASET_KEYS.items():
        if prefix in path:
            return key
    return "unknown"


test_df["dataset_key"] = test_df["image_path"].apply(assign_dataset_key)
print("\nPhân bố dataset trong test.csv:")
print(test_df["dataset_key"].value_counts())

results = {}
for key in DATASET_KEYS:
    subset = test_df[test_df["dataset_key"] == key]
    if len(subset) == 0:
        print(f"\n[SKIP] {key}: không có mẫu nào trong test.csv")
        continue
    print(f"\n{'=' * 50}")
    print(f"Dataset: {key}  ({len(subset)} mẫu)")
    print(
        f"  Real: {(subset['label'] == 'Real').sum()}  |  "
        f"Fake: {(subset['label'] == 'Fake').sum()}"
    )
    loader = DataLoader(
        FaceDataset(subset, val_transform),
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )
    metrics = evaluate(model, loader, DEVICE)
    results[key] = metrics
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")

# --- Save ---
summary = {
    k: {m: float(v) for m, v in metrics.items() if m not in ("y_true", "y_pred", "y_prob")}
    for k, metrics in results.items()
}
with open(f"{OUTPUT_DIR}/cross_generator_metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nĐã lưu: {OUTPUT_DIR}/cross_generator_metrics.json")

rows = [{"dataset": k, **m} for k, m in summary.items()]
df_summary = pd.DataFrame(rows)
df_summary.to_csv(f"{OUTPUT_DIR}/cross_generator_metrics.csv", index=False)

# Confusion matrix
for k, m in results.items():
    cm = confusion_matrix(m["y_true"], m["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {k}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/confusion_matrix_{k}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix: {k}")

# ROC curve
fig, ax = plt.subplots(figsize=(7, 6))
for k, m in results.items():
    fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
    ax.plot(fpr, tpr, label=f"{k} (AUC={m['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Cross-Generator Test")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/roc_curve_cross_generator.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved ROC curve")

# Bar chart
metrics_to_plot = ["accuracy", "f1", "auc"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
colors = ["#4C9BE8", "#E87C4C", "#4CE87C", "#E84C9B"]
for ax, metric in zip(axes, metrics_to_plot):
    vals = [summary[k][metric] for k in summary]
    keys = list(summary.keys())
    bars = ax.bar(keys, vals, color=colors[: len(keys)])
    ax.set_ylim(0.5, 1.05)
    ax.set_title(metric.upper())
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
fig.suptitle("So sánh Metric theo Dataset — Cross-Generator Test", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved metrics comparison")

# Table PNG
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.axis("off")
tbl = ax2.table(
    cellText=df_summary.round(4).values,
    colLabels=df_summary.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
fig2.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/cross_generator_table.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

print(f"\nTất cả kết quả đã lưu vào: {OUTPUT_DIR}")
print("\n=== TÓM TẮT ===")
for k, m in summary.items():
    print(f"  {k:20s}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
