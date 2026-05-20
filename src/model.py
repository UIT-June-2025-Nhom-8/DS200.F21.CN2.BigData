"""
Model definition for AI-Generated Face Detection
EfficientNet-B0 with custom classifier head
"""

import torch
import torch.nn as nn
import timm


class FaceDetectionModel(nn.Module):
    """
    EfficientNet-B0 backbone + custom binary classifier head

    Architecture:
        - Backbone: EfficientNet-B0 pretrained on ImageNet (timm)
        - Head: Linear(1280 → 256) → ReLU → Dropout(0.5) → Linear(256 → 1)
        - Output: Single logit (use BCEWithLogitsLoss)

    Args:
        backbone_name: timm model name (default: efficientnet_b0)
        pretrained: Load ImageNet pretrained weights
        hidden_dim: Hidden layer dimension in classifier head
        dropout: Dropout rate in classifier head
    """

    def __init__(
        self,
        backbone_name="efficientnet_b0",
        pretrained=True,
        hidden_dim=256,
        dropout=0.5
    ):
        super().__init__()

        # Load EfficientNet-B0 from timm
        # num_classes=0 removes the default classifier, returns features only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default head
            global_pool=""  # We'll add our own pooling
        )

        # Get feature dimension (EfficientNet-B0: 1280)
        self.feature_dim = self.backbone.num_features

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)  # Binary classification (single logit)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            logits: Output logits (B, 1) — use with BCEWithLogitsLoss
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, 1280, 7, 7)

        # Global average pooling
        features = self.global_pool(features)  # (B, 1280, 1, 1)
        features = features.flatten(1)  # (B, 1280)

        # Classifier head
        logits = self.classifier(features)  # (B, 1)

        return logits

    def freeze_backbone(self):
        """Freeze all backbone parameters (Stage 1)"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (Stage 3)"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, block_indices):
        """
        Unfreeze specific MBConv blocks in EfficientNet (Stage 2)

        Args:
            block_indices: List of block indices to unfreeze (e.g., [6, 7])
                          EfficientNet-B0 has 7 blocks (indices 0-6)
        """
        # First freeze all
        self.freeze_backbone()

        # Then unfreeze specified blocks
        for idx in block_indices:
            block_name = f"blocks.{idx}"
            for name, param in self.backbone.named_parameters():
                if name.startswith(block_name):
                    param.requires_grad = True

    def get_trainable_params(self):
        """Return list of trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(config):
    """
    Create model from config

    Args:
        config: Configuration dict loaded from YAML

    Returns:
        model: FaceDetectionModel instance
    """
    model_cfg = config["model"]

    model = FaceDetectionModel(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        hidden_dim=model_cfg["classifier"]["hidden_dim"],
        dropout=model_cfg["classifier"]["dropout"]
    )

    total_params, trainable_params = model.count_parameters()
    print(f"Model: {model_cfg['backbone']}")
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    return model
