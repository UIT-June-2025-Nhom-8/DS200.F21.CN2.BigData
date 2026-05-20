"""
Inference script - Predict Real/Fake for a single image
"""

import torch
import yaml
from pathlib import Path
import sys
import argparse
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import FaceDetectionModel


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def preprocess_image(image_path, image_size=224):
    """Load and preprocess image"""
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # Transform (same as validation set - no augmentation)
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image)
    image_tensor = transformed["image"]

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 224, 224)

    return image_tensor


def predict(model, image_tensor, device):
    """Run inference"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()

    # prob > 0.5 → Fake, prob <= 0.5 → Real
    label = "Fake" if prob > 0.5 else "Real"
    confidence = prob if prob > 0.5 else (1 - prob)

    return label, confidence


def main():
    parser = argparse.ArgumentParser(description="Predict Real/Fake for an image")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/best_stage3.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = FaceDetectionModel(
        backbone_name=config["model"]["backbone"],
        pretrained=False,  # Don't need pretrained, we'll load trained weights
        hidden_dim=config["model"]["classifier"]["hidden_dim"],
        dropout=config["model"]["classifier"]["dropout"]
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")

    # Preprocess image
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    print(f"\nProcessing: {image_path}")
    image_tensor = preprocess_image(image_path, config["data"]["image_size"])

    # Predict
    label, confidence = predict(model, image_tensor, device)

    # Print result
    print("\n" + "="*50)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
