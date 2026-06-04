"""
Inference module for FaceGuard API.
Wraps FaceDetectionModel + Grad-CAM for single-image prediction.

Label convention (from scripts/predict.py):
  sigmoid(logit) > 0.5  →  Fake
  sigmoid(logit) <= 0.5 →  Real
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT / "src"))

from model import FaceDetectionModel  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = ROOT / "artifacts/checkpoints/best_stage3.pth"
IMAGE_SIZE = 224

# ── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── Model loader ─────────────────────────────────────────────────────────────

def build_transform(model: FaceDetectionModel) -> A.Compose:
    """Build preprocessing transform using normalization stats from model.backbone.pretrained_cfg."""
    cfg = getattr(model.backbone, "pretrained_cfg", None) or getattr(model.backbone, "default_cfg", {})
    if hasattr(cfg, "mean"):
        mean, std = list(cfg.mean), list(cfg.std)
    else:
        mean = list(cfg.get("mean", (0.485, 0.456, 0.406)))
        std  = list(cfg.get("std",  (0.229, 0.224, 0.225)))
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[FaceDetectionModel, A.Compose]:
    model = FaceDetectionModel(pretrained=False)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    transform = build_transform(model)
    return model, transform

# ── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM hooked onto the last MBConv block of EfficientNet-B0."""

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

    def compute(
        self, tensor: torch.Tensor, device: torch.device, alpha: float = 0.45
    ) -> tuple[np.ndarray, float]:
        """
        Args:
            tensor : (1, 3, 224, 224) preprocessed image tensor (CPU)
            device : inference device
            alpha  : unused — kept for API compatibility

        Returns:
            cam_np : (224, 224) float32 in [0,1] — normalized CAM
            prob   : float — sigmoid(logit), probability of Fake
        """
        tensor = tensor.to(device)
        self.model.zero_grad()

        with torch.enable_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
            logits.backward()

        grads = self._gradients                          # (1, C, h, w)
        feats = self._features                           # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = cam.squeeze().cpu().numpy()                          # (h, w)

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(cam)

        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_np = np.array(cam_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)) / 255.0

        return cam_np, prob

# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess(pil_image: Image.Image, transform: A.Compose) -> tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor   : (1, 3, 224, 224) — normalized, batch-dim added
        image_np : (224, 224, 3) uint8 — resized original for overlay
    """
    rgb = np.array(pil_image.convert("RGB"))
    out = transform(image=rgb)
    tensor = out["image"].unsqueeze(0)
    image_np = np.array(
        Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    )
    return tensor, image_np

def overlay_heatmap(image_np: np.ndarray, cam_np: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = (plt.get_cmap("jet")(cam_np)[:, :, :3] * 255).astype(np.uint8)
    return np.clip(image_np * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)

def ndarray_to_b64(arr: np.ndarray) -> str:
    """Convert uint8 numpy image to base64-encoded PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

# ── Main inference function ───────────────────────────────────────────────────

def predict_image(
    pil_image: Image.Image,
    model: FaceDetectionModel,
    gradcam: GradCAM,
    device: torch.device,
    transform: A.Compose,
    include_gradcam: bool = True,
) -> dict:
    """
    Run inference + optional Grad-CAM on a single PIL image.

    Returns dict with fields matching the API response schema.
    """
    tensor, image_np = preprocess(pil_image, transform)

    t0 = time.perf_counter()

    if include_gradcam:
        cam_np, prob = gradcam.compute(tensor, device)
        overlay = overlay_heatmap(image_np, cam_np)
        gradcam_b64 = ndarray_to_b64(overlay)
    else:
        model.eval()
        with torch.no_grad():
            logits = model(tensor.to(device))
            prob = torch.sigmoid(logits).item()
        gradcam_b64 = None

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    # prob > 0.5 → Fake  (convention from scripts/predict.py)
    is_fake    = prob > 0.5
    prob_fake  = float(prob)
    prob_real  = float(1.0 - prob)
    confidence = prob_fake if is_fake else prob_real
    logit      = float(torch.log(torch.tensor(prob / (1 - prob + 1e-8))).item())

    return {
        "label":         "GIẢ (AI)" if is_fake else "THẬT",
        "label_en":      "Fake"     if is_fake else "Real",
        "is_fake":       is_fake,
        "prob_fake":     round(prob_fake, 4),
        "prob_real":     round(prob_real, 4),
        "confidence":    round(confidence, 4),
        "logit":         round(logit, 4),
        "inference_ms":  inference_ms,
        "gradcam_b64":   gradcam_b64,
    }
