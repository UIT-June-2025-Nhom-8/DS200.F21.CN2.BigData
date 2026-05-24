"""
Standalone evaluation script
Use this to evaluate a trained model checkpoint on test set
"""

import yaml
import torch
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import create_dataloaders
from model import create_model
from evaluate import Evaluator


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
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
    config_path = Path(args.config)
    config = load_config(config_path)

    print("="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    print(f"Config    : {config_path}")
    print(f"Checkpoint: {args.checkpoint}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")

    # Load test dataloader
    print("\n" + "-"*70)
    print("Loading test dataset...")
    print("-"*70)
    _, _, test_loader = create_dataloaders(config)

    # Create model
    print("\n" + "-"*70)
    print("Loading model...")
    print("-"*70)
    model = create_model(config)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    val_loss = checkpoint.get('val_loss')
    val_acc = checkpoint.get('val_acc')

    if isinstance(val_loss, (int, float)):
        print(f"  Val Loss: {val_loss:.4f}")
    else:
        print(f"  Val Loss: N/A")

    if isinstance(val_acc, (int, float)):
        print(f"  Val Acc : {val_acc:.4f}")
    else:
        print(f"  Val Acc : N/A")

    # Evaluate
    results_dir = Path(config["paths"]["results_dir"])
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        results_dir=results_dir
    )

    metrics = evaluator.evaluate()

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
