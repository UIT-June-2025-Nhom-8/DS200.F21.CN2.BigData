"""
Training engine for 3-stage progressive unfreezing
Supports mixed precision, callbacks, and checkpointing
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
from tqdm import tqdm


class Trainer:
    """
    3-stage training engine with callbacks

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Configuration dict
        device: torch.device
    """

    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Mixed precision scaler
        self.use_amp = config["training"]["mixed_precision"]
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Checkpoints directory
        self.ckpt_dir = Path(config["paths"]["checkpoints_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "stage1": {"train_loss": [], "val_loss": [], "val_acc": []},
            "stage2": {"train_loss": [], "val_loss": [], "val_acc": []},
            "stage3": {"train_loss": [], "val_loss": [], "val_acc": []}
        }

    def train_stage(self, stage_name, stage_config, stage_num):
        """
        Train a single stage

        Args:
            stage_name: Stage identifier (e.g., "stage1")
            stage_config: Stage configuration dict
            stage_num: Stage number (1, 2, or 3)

        Returns:
            best_checkpoint_path: Path to best model checkpoint
        """
        print(f"\n{'='*70}")
        print(f"STAGE {stage_num}: {stage_config['name']}")
        print(f"{'='*70}")

        # Configure model freezing
        if stage_config["freeze_backbone"]:
            self.model.freeze_backbone()
            print("✓ Backbone frozen")
        elif stage_config["unfreeze_blocks"] == "all":
            self.model.unfreeze_backbone()
            print("✓ All layers unfrozen")
        else:
            self.model.unfreeze_top_blocks(stage_config["unfreeze_blocks"])
            print(f"✓ Unfrozen blocks: {stage_config['unfreeze_blocks']}")

        # Count trainable parameters
        total_params, trainable_params = self.model.count_parameters()
        print(f"Trainable params: {trainable_params:,} / {total_params:,}")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.get_trainable_params(),
            lr=stage_config["learning_rate"],
            weight_decay=self.config["training"]["optimizer"]["weight_decay"]
        )

        # Learning rate scheduler (ReduceLROnPlateau)
        scheduler_cfg = self.config["callbacks"]["reduce_lr_on_plateau"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg["mode"],
            patience=scheduler_cfg["patience"],
            factor=scheduler_cfg["factor"],
            min_lr=scheduler_cfg["min_lr"]
        )

        # Early stopping
        es_cfg = self.config["callbacks"]["early_stopping"]
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        # Training loop
        max_epochs = self.config["training"]["max_epochs_per_stage"]

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(optimizer)

            # Validate
            val_loss, val_acc = self._validate_epoch()

            # Update history
            self.history[stage_name]["train_loss"].append(train_loss)
            self.history[stage_name]["val_loss"].append(val_loss)
            self.history[stage_name]["val_acc"].append(val_acc)

            # Learning rate scheduler step
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:2d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Model checkpoint (save best)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                # Save checkpoint
                ckpt_path = self.ckpt_dir / f"best_{stage_name}.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "config": self.config
                }, ckpt_path)
                print(f"✓ Saved best checkpoint: {ckpt_path}")

            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= es_cfg["patience"]:
                print(f"\n⚠ Early stopping triggered (patience={es_cfg['patience']})")
                print(f"Best epoch: {best_epoch} | Best val_loss: {best_val_loss:.4f}")
                break

        # Load best checkpoint for next stage
        best_ckpt_path = self.ckpt_dir / f"best_{stage_name}.pth"
        print(f"\n✓ Stage {stage_num} complete. Best checkpoint: {best_ckpt_path}")

        return best_ckpt_path

    def _train_epoch(self, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # (B,) → (B, 1)

            optimizer.zero_grad()

            # Mixed precision forward pass
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                # Backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)  # (B,) → (B, 1)

                # Forward pass
                if self.use_amp:
                    with autocast('cuda'):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()

                # Accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train_all_stages(self):
        """
        Train all 3 stages sequentially

        Returns:
            final_checkpoint_path: Path to best model from Stage 3
        """
        training_cfg = self.config["training"]

        # Stage 1
        stage1_ckpt = self.train_stage("stage1", training_cfg["stage1"], stage_num=1)

        # Load Stage 1 checkpoint for Stage 2
        checkpoint = torch.load(stage1_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n✓ Loaded Stage 1 checkpoint for Stage 2")

        # Stage 2
        stage2_ckpt = self.train_stage("stage2", training_cfg["stage2"], stage_num=2)

        # Load Stage 2 checkpoint for Stage 3
        checkpoint = torch.load(stage2_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n✓ Loaded Stage 2 checkpoint for Stage 3")

        # Stage 3
        stage3_ckpt = self.train_stage("stage3", training_cfg["stage3"], stage_num=3)

        print(f"\n{'='*70}")
        print("ALL STAGES COMPLETE")
        print(f"{'='*70}")
        print(f"Final model: {stage3_ckpt}")

        return stage3_ckpt
