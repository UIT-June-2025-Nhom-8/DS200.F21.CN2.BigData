"""
Evaluation metrics and visualization for test set
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


class Evaluator:
    """
    Evaluate model on test set and generate metrics + visualizations

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: torch.device
        results_dir: Directory to save results
    """

    def __init__(self, model, test_loader, device, results_dir):
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def predict(self):
        """
        Run inference on test set

        Returns:
            y_true: Ground truth labels (numpy array)
            y_pred: Predicted labels (numpy array)
            y_proba: Predicted probabilities (numpy array)
        """
        self.model.eval()

        all_labels = []
        all_probs = []

        for images, labels in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)

            # Forward pass
            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels.numpy())
            all_probs.append(probs)

        # Concatenate batches
        y_true = np.concatenate(all_labels)
        y_proba = np.concatenate(all_probs).flatten()
        y_pred = (y_proba > 0.5).astype(int)

        return y_true, y_pred, y_proba

    def compute_metrics(self, y_true, y_pred, y_proba):
        """
        Compute classification metrics

        Returns:
            metrics: Dict of metric values
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc_roc": roc_auc_score(y_true, y_proba)
        }

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            cbar_kws={"label": "Count"}
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        save_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved confusion matrix: {save_path}")

    def plot_roc_curve(self, y_true, y_proba, auc_score):
        """Plot and save ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = self.results_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved ROC curve: {save_path}")

    def evaluate(self):
        """
        Run full evaluation pipeline

        Returns:
            metrics: Dict of metric values
        """
        print("\n" + "="*70)
        print("EVALUATION ON TEST SET")
        print("="*70)

        # Predict
        y_true, y_pred, y_proba = self.predict()

        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_proba)

        # Print metrics
        print("\nMetrics:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1']:.4f}")
        print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)

        # Plot ROC curve
        self.plot_roc_curve(y_true, y_proba, metrics["auc_roc"])

        # Save metrics to file
        metrics_path = self.results_dir / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write("Test Set Metrics\n")
            f.write("="*50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key:12s}: {value:.4f}\n")

        print(f"\n✓ Saved metrics: {metrics_path}")

        return metrics


def plot_training_curves(history, results_dir):
    """
    Plot training/validation loss and accuracy curves for all stages

    Args:
        history: Training history dict from Trainer
        results_dir: Directory to save plots
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    stages = ["stage1", "stage2", "stage3"]
    stage_names = ["Stage 1: Freeze Backbone", "Stage 2: Unfreeze Top Blocks", "Stage 3: Unfreeze All"]

    for i, (stage, stage_name) in enumerate(zip(stages, stage_names)):
        if stage not in history or len(history[stage]["train_loss"]) == 0:
            continue

        # Loss plot
        ax_loss = axes[0, i]
        epochs = range(1, len(history[stage]["train_loss"]) + 1)
        ax_loss.plot(epochs, history[stage]["train_loss"], label="Train Loss", marker="o")
        ax_loss.plot(epochs, history[stage]["val_loss"], label="Val Loss", marker="s")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{stage_name}\nLoss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)

        # Accuracy plot
        ax_acc = axes[1, i]
        ax_acc.plot(epochs, history[stage]["val_acc"], label="Val Accuracy", marker="o", color="#4CAF50")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(f"{stage_name}\nValidation Accuracy")
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    save_path = results_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved training curves: {save_path}")
