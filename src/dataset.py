"""
Dataset loader for AI-Generated Face Detection
Supports on-the-fly augmentation for training set
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceDataset(Dataset):
    """
    PyTorch Dataset for Real/Fake face images

    Args:
        csv_path: Path to CSV file (columns: image_path, label)
        image_size: Target image size (default: 224)
        mean: ImageNet mean for normalization
        std: ImageNet std for normalization
        augment: Whether to apply augmentation (train set only)
        augment_config: Augmentation parameters from config
    """

    def __init__(
        self,
        csv_path,
        image_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        augment=False,
        augment_config=None
    ):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment

        # Label mapping: Real=0, Fake=1
        self.label_map = {"Real": 0, "Fake": 1}

        # Build transforms
        self.transform = self._build_transform(augment, augment_config)

    def _build_transform(self, augment, augment_config):
        """Build Albumentations transform pipeline"""

        transforms_list = []

        # Always resize
        transforms_list.append(A.Resize(self.image_size, self.image_size))

        # Augmentation for training set
        if augment and augment_config is not None:
            # HorizontalFlip
            if "horizontal_flip" in augment_config:
                transforms_list.append(
                    A.HorizontalFlip(p=augment_config["horizontal_flip"]["p"])
                )

            # RandomBrightnessContrast
            if "random_brightness_contrast" in augment_config:
                cfg = augment_config["random_brightness_contrast"]
                transforms_list.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=cfg["brightness_limit"],
                        contrast_limit=cfg["contrast_limit"],
                        p=cfg["p"]
                    )
                )

            # ShiftScaleRotate
            if "shift_scale_rotate" in augment_config:
                cfg = augment_config["shift_scale_rotate"]
                transforms_list.append(
                    A.ShiftScaleRotate(
                        shift_limit=cfg["shift_limit"],
                        scale_limit=cfg["scale_limit"],
                        rotate_limit=cfg["rotate_limit"],
                        border_mode=0,  # cv2.BORDER_CONSTANT
                        p=cfg["p"]
                    )
                )

            # CoarseDropout
            if "coarse_dropout" in augment_config:
                cfg = augment_config["coarse_dropout"]
                transforms_list.append(
                    A.CoarseDropout(
                        max_holes=cfg["max_holes"],
                        max_height=cfg["max_height"],
                        max_width=cfg["max_width"],
                        fill_value=0,
                        p=cfg["p"]
                    )
                )

        # Normalize (ImageNet mean/std)
        transforms_list.append(A.Normalize(mean=self.mean, std=self.std))

        # Convert to PyTorch tensor
        transforms_list.append(ToTensorV2())

        return A.Compose(transforms_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"[ERROR] Failed to load {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        # Get label
        label_str = row["label"]
        label = self.label_map[label_str]
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor


def create_dataloaders(config):
    """
    Create train, val, test DataLoaders from config

    Args:
        config: Configuration dict loaded from YAML

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    data_cfg = config["data"]
    aug_cfg = config.get("augmentation", None)

    splits_dir = Path(data_cfg["splits_dir"])

    # Train dataset (with augmentation)
    train_dataset = FaceDataset(
        csv_path=splits_dir / data_cfg["train_csv"],
        image_size=data_cfg["image_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        augment=True,
        augment_config=aug_cfg
    )

    # Val dataset (no augmentation)
    val_dataset = FaceDataset(
        csv_path=splits_dir / data_cfg["val_csv"],
        image_size=data_cfg["image_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        augment=False,
        augment_config=None
    )

    # Test dataset (no augmentation)
    test_dataset = FaceDataset(
        csv_path=splits_dir / data_cfg["test_csv"],
        image_size=data_cfg["image_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        augment=False,
        augment_config=None
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True if data_cfg["num_workers"] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True if data_cfg["num_workers"] > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True if data_cfg["num_workers"] > 0 else False
    )

    print(f"Train: {len(train_dataset):,} images")
    print(f"Val  : {len(val_dataset):,} images")
    print(f"Test : {len(test_dataset):,} images")

    return train_loader, val_loader, test_loader
