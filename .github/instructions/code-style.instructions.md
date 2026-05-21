---
applyTo: 'src/**/*.py,scripts/**/*.py,notebooks/**/*.py,notebooks/**/*.ipynb'
---

# Code Style — DS200 AI Face Detection Project

## PyTorch Conventions

- Always use `torch.nn.Module` subclasses for model definitions; never use bare `nn.Sequential` at the top level.
- Place `model.train()` / `model.eval()` explicitly before training and evaluation loops.
- Use `torch.cuda.amp.autocast()` and `torch.cuda.amp.GradScaler` for mixed-precision training; do not disable without a comment explaining why.
- Move tensors to device with `.to(device)` — define `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` once at the top of each script.
- Never hardcode `cuda:0`; always use the `device` variable.

## timm Conventions

- Create models with `timm.create_model(model_name, pretrained=True, num_classes=0)` then attach a custom head — do not pass `num_classes` directly when using a custom classifier.
- Store the `model_name` string in config (YAML or argparse), never hardcode in training logic.
- Use `model.default_cfg` to retrieve the correct ImageNet normalization stats instead of hardcoding mean/std values.

## Dataset & DataLoader

- All `torch.utils.data.Dataset` subclasses live in `src/dataset.py` (or `src/data/`).
- `__getitem__` must return `(image_tensor, label_tensor)` — labels as `torch.float32` for BCE loss.
- Augmentation pipelines use `albumentations`; define `train_transform` and `val_transform` separately. Never augment val/test sets beyond resize + normalize.
- Set `num_workers=4` and `pin_memory=True` on DataLoaders when GPU is available.

## Training Loop

- Loss: `torch.nn.BCEWithLogitsLoss` — do not apply `sigmoid` before the loss.
- Optimizer: `torch.optim.AdamW` with `weight_decay=1e-4`.
- Checkpoint: save `{"epoch": epoch, "model_state_dict": ..., "optimizer_state_dict": ..., "val_loss": ..., "val_acc": ...}` — never save the full model object.
- Log per-epoch metrics with a structured dict, not bare `print` statements. Use `tqdm` for progress bars.

## Evaluation

- Always evaluate on the **test set**, never on the validation set for final reported numbers.
- Compute metrics using `sklearn.metrics`: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`.
- Save confusion matrix and ROC curve figures to `reports/` — never inline in notebooks as the only copy.

## File & Import Organization

- Imports order: stdlib → third-party → local (`src/`). Separated by blank lines.
- No wildcard imports (`from module import *`).
- Config values (paths, hyperparameters) come from YAML files in `configs/` or CLI args — never hardcoded in `src/`.
- Artifact paths (checkpoints, figures) reference `artifacts/` and `reports/` relative to the project root.

## Naming

| Thing               | Convention                 | Example                                 |
| ------------------- | -------------------------- | --------------------------------------- |
| Classes             | `PascalCase`               | `FaceDataset`, `EfficientNetClassifier` |
| Functions / methods | `snake_case`               | `train_one_epoch`, `compute_metrics`    |
| Constants           | `UPPER_SNAKE_CASE`         | `IMAGENET_MEAN`, `NUM_CLASSES`          |
| Config keys (YAML)  | `snake_case`               | `learning_rate`, `batch_size`           |
| Checkpoint files    | `{model}_{stage}_best.pth` | `efficientnet_b0_stage1_best.pth`       |

## Reproducibility

- Set seeds at the start of every training script:
  ```python
  import random, numpy as np, torch
  random.seed(42); np.random.seed(42); torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
  ```
- Log `torch.__version__`, `timm.__version__`, and GPU name to the run summary.
