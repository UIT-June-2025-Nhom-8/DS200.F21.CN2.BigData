"""
Main training script
Run 3-stage training pipeline from config
"""

import yaml
import torch
import random
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import create_dataloaders
from model import create_model
from trainer import Trainer
from evaluate import Evaluator, plot_training_curves


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "train_config.yaml"
    config = load_config(config_path)

    print("="*70)
    print("AI-GENERATED FACE DETECTION - TRAINING")
    print("="*70)
    print(f"Config: {config_path}")

    # Set seed
    set_seed(config["seed"])
    print(f"Random seed: {config['seed']}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataloaders
    print("\n" + "-"*70)
    print("Loading datasets...")
    print("-"*70)
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create model
    print("\n" + "-"*70)
    print("Creating model...")
    print("-"*70)
    model = create_model(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train all 3 stages
    print("\n" + "-"*70)
    print("Starting 3-stage training...")
    print("-"*70)
    final_checkpoint = trainer.train_all_stages()

    # Plot training curves
    print("\n" + "-"*70)
    print("Plotting training curves...")
    print("-"*70)
    results_dir = Path(config["paths"]["results_dir"])
    plot_training_curves(trainer.history, results_dir)

    # Evaluate on test set
    print("\n" + "-"*70)
    print("Evaluating on test set...")
    print("-"*70)

    # Load best model
    checkpoint = torch.load(final_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        results_dir=results_dir
    )

    metrics = evaluator.evaluate()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best model: {final_checkpoint}")
    print(f"Results saved to: {results_dir}")
    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1']:.4f}")
    print(f"Test AUC-ROC : {metrics['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
