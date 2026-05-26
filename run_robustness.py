"""Chạy thực nghiệm Robustness và lưu kết quả vào tests/robustness/."""
import io
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
OUTPUT_DIR = "reports/robustness"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
        f"Loaded checkpoint -- epoch {ckpt['epoch']}, "
        f"val_loss {ckpt['val_loss']:.4f}, val_acc {ckpt['val_acc']:.4f}"
    )
    model.eval()
    return model


def apply_jpeg_compression(img_pil, quality):
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_resize_downup(img_pil, scale):
    w, h = img_pil.size
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small = img_pil.resize((small_w, small_h), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def apply_gaussian_blur(img_pil, sigma):
    return img_pil.filter(ImageFilter.GaussianBlur(radius=sigma))


PERTURBATIONS = {
    "baseline": lambda img: img,
    "jpeg_q70": lambda img: apply_jpeg_compression(img, 70),
    "jpeg_q50": lambda img: apply_jpeg_compression(img, 50),
    "jpeg_q30": lambda img: apply_jpeg_compression(img, 30),
    "resize_50": lambda img: apply_resize_downup(img, 0.50),
    "resize_25": lambda img: apply_resize_downup(img, 0.25),
    "blur_s1": lambda img: apply_gaussian_blur(img, 1.0),
    "blur_s2": lambda img: apply_gaussian_blur(img, 2.0),
}

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


class RobustnessDataset(Dataset):
    def __init__(self, df, perturb_fn, transform):
        self.df = df.reset_index(drop=True)
        self.perturb_fn = perturb_fn
        self.transform = transform
        self.label_map = {"Real": 0, "Fake": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = self.perturb_fn(img)
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img_np = np.array(img)
        img_t = self.transform(image=img_np)["image"]
        label = torch.tensor(self.label_map[row["label"]], dtype=torch.float32)
        return img_t, label


model = load_model(CHECKPOINT_PATH, DEVICE)
test_df = pd.read_csv(TEST_CSV)
print(f"Test set: {len(test_df)} samples")

robustness_results = {}

for name, perturb_fn in PERTURBATIONS.items():
    print(f"\n[{name}] evaluating...", flush=True)
    dataset = RobustnessDataset(test_df, perturb_fn, val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    all_preds, all_probs, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            logits = model(imgs.to(DEVICE)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            if (batch_idx + 1) % 50 == 0:
                print(f"  batch {batch_idx + 1}/{len(loader)}", flush=True)

    y = np.array(all_labels)
    p = np.array(all_preds)
    prob = np.array(all_probs)
    robustness_results[name] = {
        "accuracy": float(accuracy_score(y, p)),
        "auc": float(roc_auc_score(y, prob)),
    }
    print(
        f"  Accuracy={robustness_results[name]['accuracy']:.4f}  "
        f"AUC={robustness_results[name]['auc']:.4f}",
        flush=True,
    )

with open(f"{OUTPUT_DIR}/robustness_metrics.json", "w", encoding="utf-8") as f:
    json.dump(robustness_results, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/robustness_metrics.json")

ORDER = ["baseline", "jpeg_q70", "jpeg_q50", "jpeg_q30",
         "resize_50", "resize_25", "blur_s1", "blur_s2"]
names = [n for n in ORDER if n in robustness_results]
acc_vals = [robustness_results[n]["accuracy"] for n in names]
auc_vals = [robustness_results[n]["auc"] for n in names]
baseline_acc = robustness_results["baseline"]["accuracy"]
baseline_auc = robustness_results["baseline"]["auc"]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(names, acc_vals, marker="o", linewidth=2, color="#4C9BE8", label="Accuracy")
ax.axhline(baseline_acc, linestyle="--", color="#999", linewidth=1,
           label=f"Baseline ({baseline_acc:.3f})")
for n, v in zip(names, acc_vals):
    ax.annotate(f"{v:.3f}", (n, v),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.set_ylim(max(0, min(acc_vals) - 0.05), 1.02)
ax.set_xlabel("Perturbation")
ax.set_ylabel("Accuracy")
ax.set_title("Robustness Test -- Accuracy by Perturbation")
ax.legend()
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/robustness_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved robustness_accuracy.png")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(names, auc_vals, marker="s", linewidth=2, color="#E87C4C", label="AUC-ROC")
ax.axhline(baseline_auc, linestyle="--", color="#999", linewidth=1,
           label=f"Baseline ({baseline_auc:.3f})")
for n, v in zip(names, auc_vals):
    ax.annotate(f"{v:.3f}", (n, v),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)
ax.set_ylim(max(0, min(auc_vals) - 0.05), 1.02)
ax.set_xlabel("Perturbation")
ax.set_ylabel("AUC-ROC")
ax.set_title("Robustness Test -- AUC-ROC by Perturbation")
ax.legend()
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/robustness_auc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved robustness_auc.png")

acc_drops = [baseline_acc - v for v in acc_vals]
colors = ["#4CE87C" if d <= 0.02 else "#E8C44C" if d <= 0.05 else "#E84C4C"
          for d in acc_drops]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(names, acc_drops, color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Perturbation")
ax.set_ylabel("Accuracy Drop (vs baseline)")
ax.set_title("Robustness Test -- Accuracy Drop")
ax.tick_params(axis="x", rotation=20)
for bar, v in zip(bars, acc_drops):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/robustness_drop.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved robustness_drop.png")

print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\n=== SUMMARY ===")
for name in names:
    m = robustness_results[name]
    drop = baseline_acc - m["accuracy"]
    print(f"  {name:12s}  Acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}  Drop={drop:+.4f}")
