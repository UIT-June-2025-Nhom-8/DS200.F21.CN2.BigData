"""
Quick test script to verify setup before training
Tests: config loading, data loading, model creation, forward pass
"""

import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset import create_dataloaders
from model import create_model


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def test_setup():
    print("="*70)
    print("SETUP VERIFICATION TEST")
    print("="*70)

    # 1. Test config loading
    print("\n[1/5] Testing config loading...")
    try:
        config_path = Path(__file__).parent.parent / "configs" / "train_config.yaml"
        config = load_config(config_path)
        print(f"✓ Config loaded from {config_path}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False

    # 2. Test CUDA availability
    print("\n[2/5] Testing CUDA availability...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print(f"⚠ CUDA not available, using CPU (training will be very slow)")

    # 3. Test data loading
    print("\n[3/5] Testing data loading...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        print(f"✓ DataLoaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches  : {len(val_loader)}")
        print(f"  Test batches : {len(test_loader)}")
    except Exception as e:
        print(f"✗ Failed to create dataloaders: {e}")
        return False

    # 4. Test model creation
    print("\n[4/5] Testing model creation...")
    try:
        model = create_model(config)
        model = model.to(device)
        print(f"✓ Model created and moved to {device}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False

    # 5. Test forward pass
    print("\n[5/5] Testing forward pass...")
    try:
        # Get one batch
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        model.eval()
        with torch.no_grad():
            logits = model(images)

        print(f"✓ Forward pass successful")
        print(f"  Input shape : {images.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Labels shape: {labels.shape}")

        # Test loss computation
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, labels.unsqueeze(1))
        print(f"  Loss value  : {loss.item():.4f}")

    except Exception as e:
        print(f"✗ Failed forward pass: {e}")
        return False

    # All tests passed
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - READY TO TRAIN")
    print("="*70)
    print("\nTo start training, run:")
    print("  python scripts/run_train.py")
    print("\nEstimated training time: 3-5 hours on RTX 4070/5070")

    return True


if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
