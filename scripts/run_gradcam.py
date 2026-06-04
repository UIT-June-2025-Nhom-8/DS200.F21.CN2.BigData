"""
Grad-CAM visualization for AI-Generated Face Detection
Generates heatmap overlays for sample images from each dataset source.

Target layer : last MBConv block of EfficientNet-B0 backbone
Output       : artifacts/gradcam/  (individual PNGs + summary grid)

Usage:
    python scripts/run_gradcam.py
    python scripts/run_gradcam.py --n_per_class 4
    python scripts/run_gradcam.py --checkpoint artifacts/checkpoints/best_stage3.pth
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "src"))

from model import FaceDetectionModel  # noqa: E402

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SOURCE_MAP = {
    "140k-real-and-fake-faces": "140k-StyleGAN",
    "deepfake-and-real-images":  "Deepfake-Real",
    "hardfakevsrealfaces":        "Hard-FakeReal",
    "real-and-fake-face-detection": "ciplab",
}


# ---------------------------------------------------------------------------
# Device selection — M1-compatible
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS backward support is stable from PyTorch 2.x on Apple Silicon
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> FaceDetectionModel:
    model = FaceDetectionModel(pretrained=False)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        # weights_only kwarg not available in older PyTorch
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc")
    val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else "N/A"
    print(f"Checkpoint : {checkpoint_path.name}  (epoch {epoch}, val_acc {val_acc_str})")
    return model


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def load_image(image_path: str):
    """Return (tensor (1,3,224,224), numpy uint8 (224,224,3))."""
    with Image.open(image_path) as img:
        image_np_orig = np.array(img.convert("RGB"))
    transformed = TRANSFORM(image=image_np_orig)
    tensor = transformed["image"].unsqueeze(0)   # (1, 3, 224, 224)
    # Also return the resized-but-not-normalized version for display
    image_np = np.array(
        Image.fromarray(image_np_orig).resize((224, 224), Image.BILINEAR)
    )
    return tensor, image_np


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM using PyTorch forward/backward hooks.
    Hooks are attached to model.backbone.blocks[-1]
    (last MBConv block of EfficientNet-B0).
    """

    def __init__(self, model: FaceDetectionModel):
        self.model = model
        self._features: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._attach_hooks()

    def _attach_hooks(self):
        target = self.model.backbone.blocks[-1]

        def _fwd(module, inp, out):
            self._features = out.detach()

        def _bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        target.register_forward_hook(_fwd)
        target.register_full_backward_hook(_bwd)

    def __call__(self, tensor: torch.Tensor, device: torch.device):
        """
        Args:
            tensor : (1, 3, 224, 224) — NOT on device yet
        Returns:
            cam  : (224, 224) float32 numpy in [0, 1]
            prob : float — probability of Fake
        """
        tensor = tensor.to(device)
        self.model.zero_grad()

        with torch.enable_grad():
            logits = self.model(tensor)           # (1, 1)
            prob = torch.sigmoid(logits).item()
            logits.backward()

        # weights: global average of gradients over spatial dims
        grads = self._gradients                   # (1, C, h, w)
        feats = self._features                    # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = cam.squeeze().cpu().numpy()         # (h, w)

        # Normalize to [0, 1]
        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)

        # Upsample to 224×224
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam = np.array(cam_img.resize((224, 224), Image.BILINEAR)) / 255.0

        return cam, prob


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend jet heatmap onto original image."""
    heatmap = (plt.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.uint8)
    return np.clip(image_np * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def infer_source(path_str: str) -> str:
    for key, name in SOURCE_MAP.items():
        if key in path_str:
            return name
    return "Unknown"


def sample_images(test_csv: Path, n_per_class: int):
    df = pd.read_csv(test_csv)
    df["source"] = df["image_path"].apply(infer_source)

    rows = []
    for source_name in SOURCE_MAP.values():
        src_df = df[df["source"] == source_name]
        for label in ["Real", "Fake"]:
            subset = src_df[src_df["label"] == label]
            if subset.empty:
                print(f"  [WARN] No {label} images for source: {source_name}")
                continue
            n = min(n_per_class, len(subset))
            chosen = subset.sample(n=n, random_state=42)
            for _, row in chosen.iterrows():
                rows.append({
                    "image_path": row["image_path"],
                    "label": label,
                    "source": source_name,
                })
    return rows


# ---------------------------------------------------------------------------
# Grid figure
# ---------------------------------------------------------------------------

def save_grid(results: list, output_dir: Path):
    n = len(results)
    if n == 0:
        return

    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 3.0))
    axes = np.array(axes).flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        ax.imshow(r["overlay"])
        correct = r["pred"] == r["label"]
        color = "limegreen" if correct else "tomato"
        ax.set_title(
            f"{r['source']}\n"
            f"GT: {r['label']}  Pred: {r['pred']}\n"
            f"p(Fake)={r['prob']:.3f}",
            fontsize=5.5, color=color, pad=2
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Grad-CAM — EfficientNet-B0 Stage 3\n"
        "Màu viền tiêu đề: xanh lá = dự đoán đúng, đỏ = dự đoán sai",
        fontsize=8, y=1.01
    )
    plt.tight_layout()
    out = output_dir / "gradcam_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate Grad-CAM overlays")
    p.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts/checkpoints/best_stage3.pth"),
        help="Path to model checkpoint (default: best_stage3.pth)",
    )
    p.add_argument(
        "--test_csv",
        default=str(ROOT / "data/splits/test.csv"),
        help="Path to test split CSV",
    )
    p.add_argument(
        "--output_dir",
        default=str(ROOT / "artifacts/gradcam"),
        help="Directory to save Grad-CAM outputs",
    )
    p.add_argument(
        "--n_per_class",
        type=int,
        default=4,
        help="Number of Real and Fake samples per source (default: 4)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay opacity (default: 0.45)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device     : {device}")

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = load_model(ckpt_path, device)

    # Init Grad-CAM
    gradcam = GradCAM(model)

    # Sample images
    test_csv = Path(args.test_csv)
    if not test_csv.exists():
        print(f"[ERROR] test.csv not found: {test_csv}")
        sys.exit(1)
    samples = sample_images(test_csv, n_per_class=args.n_per_class)
    print(f"Samples    : {len(samples)} images ({args.n_per_class} Real + {args.n_per_class} Fake × 4 sources)")

    results = []
    for s in samples:
        img_path = s["image_path"]
        if not Path(img_path).exists():
            print(f"  [SKIP] File not found: {img_path}")
            continue
        try:
            tensor, image_np = load_image(img_path)
            cam, prob = gradcam(tensor, device)
            overlay = overlay_heatmap(image_np, cam, alpha=args.alpha)

            pred = "Fake" if prob > 0.5 else "Real"
            result = {
                "image_np": image_np,
                "overlay":  overlay,
                "label":    s["label"],
                "pred":     pred,
                "prob":     prob,
                "source":   s["source"],
                "path":     img_path,
            }
            results.append(result)

            # Save individual overlay
            stem = Path(img_path).stem
            fname = f"{s['source'].replace(' ', '_')}_{s['label']}_{stem}.png"
            Image.fromarray(overlay).save(output_dir / fname)

        except Exception as e:
            print(f"  [ERROR] {img_path}: {e}")
            continue

    print(f"\nGenerated  : {len(results)} overlays → {output_dir}")

    # Save summary grid
    save_grid(results, output_dir)

    # Print accuracy summary
    if results:
        correct = sum(1 for r in results if r["pred"] == r["label"])
        print(f"Accuracy on sampled images: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
        print("\nPer-source summary:")
        for src in SOURCE_MAP.values():
            src_res = [r for r in results if r["source"] == src]
            if src_res:
                src_correct = sum(1 for r in src_res if r["pred"] == r["label"])
                print(f"  {src:<20}: {src_correct}/{len(src_res)} correct")


if __name__ == "__main__":
    main()
