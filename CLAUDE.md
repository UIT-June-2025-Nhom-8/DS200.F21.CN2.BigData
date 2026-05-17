# DS200.F21.CN2 — AI Face Detection

> **Course:** DS200.F21.CN2, UIT-VNU HCM  
> **Task:** Binary classification of real vs. AI-generated face images using EfficientNet-B0.

---

## Language Policy

**All human-readable outputs must be in Vietnamese.** Technical terms (model names, metric names, library/package names, variable names, CLI flags) may remain in English.

This applies to:

- Academic report and all its sections
- EDA reports and figure titles/captions
- Training summaries and progress reports
- Preprocessing reports
- Test plans, test cases, and test result summaries
- Any Markdown or text file written to `reports/`

Do **not** translate: code, file paths, metric keys (`val_loss`, `AUC`), library names (`albumentations`, `timm`), or CLI output from scripts.

---

## Project Structure

```
artifacts/        # Checkpoints (efficientnet_b0_stage{1,2,3}_best.pth), Grad-CAM overlays
configs/          # YAML configs for training/eval/inference
data/
  raw/            # Original Kaggle datasets (do not modify)
  splits/         # train.csv, val.csv, test.csv  — columns: image_path, label
notebooks/        # Notebooks area; planned names may include 00_download_datasets.ipynb, 01_dataset_preparation.ipynb
reports/          # EDA figures, eval metrics, confusion matrix, ROC curve
  scripts/          # Scaffolded utility-scripts directory (currently placeholder only); planned scripts include prepare_data.py, demo.py
  src/              # Scaffolded source directory (currently placeholder only); planned modules include dataset.py, model.py, train.py, eval.py
tests/            # Unit and integration tests
```

> **Note:** `scripts/` and `src/` are currently empty (`.gitkeep` placeholders). The modules listed above are planned and will be added during the implementation phase.

---

## Datasets

Four Kaggle datasets, all under `data/raw/`:

| Key           | Path under data/raw/                                 | Size                       |
| ------------- | ---------------------------------------------------- | -------------------------- |
| 140k-StyleGAN | `140k-real-and-fake-faces/real_vs_fake/real-vs-fake` | 70k real + 70k fake        |
| Deepfake-Real | `deepfake-and-real-images/Dataset`                   | 256×256 manipulation-based |
| Hard-FakeReal | `hardfakevsrealfaces`                                | Small, hard to classify    |
| ciplab        | `real-and-fake-face-detection/real_and_fake_face`    | Mixed sources              |

Labels are inferred from directory names:

- Real: `real`, `Real`, `training_real`
- Fake: `fake`, `Fake`, `training_fake`

Split: **70% train / 15% val / 15% test**, stratified, `random_state=42`.  
CSV schema: `image_path` (absolute str), `label` (int: 0=fake, 1=real).

---

## Data Pipeline

Run notebooks in order when `data/splits/` is empty:

1. **`notebooks/00_download_datasets.ipynb`** — Downloads all 4 Kaggle datasets. Idempotent (skips existing). Requires Kaggle credentials filled in cell 1. Run cleanup cell after.
2. **`notebooks/01_dataset_preparation.ipynb`** — MD5 dedup → class balance check → EDA plots → 70/15/15 split → saves CSVs. Outputs EDA figures to `reports/`.

---

## Model

- **Architecture:** `timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)` + custom head: `Linear(1280,256) → ReLU → Dropout(0.5) → Linear(256,1)`
- **Loss:** `BCEWithLogitsLoss` (do NOT apply sigmoid before loss)
- **Optimizer:** `AdamW`, `weight_decay=1e-4`
- **Mixed precision:** `torch.cuda.amp.autocast()` + `GradScaler`
- **Device:** `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` — never hardcode `cuda:0`

---

## Training Stages

| Stage | Trainable layers               | LR   | Max epochs | Stop condition                    |
| ----- | ------------------------------ | ---- | ---------- | --------------------------------- |
| 1     | Head only (backbone frozen)    | 1e-4 | 20–30      | EarlyStopping patience=5          |
| 2     | Head + EfficientNet blocks 6–7 | 1e-5 | 20–30      | EarlyStopping patience=5          |
| 3     | Full model                     | 1e-6 | 20–30      | EarlyStopping + ReduceLROnPlateau |

- Batch size: 128
- Save best checkpoint per stage to `artifacts/` keyed by `val_loss`
- Checkpoint format: `{"epoch", "model_state_dict", "optimizer_state_dict", "val_loss", "val_acc"}` — never save the full model object

---

## Evaluation

- **Final metrics always on test set**, never val set
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC (via `sklearn.metrics`)
- Save confusion matrix + ROC curve to `reports/`
- **Cross-generator test:** hold out one dataset, train on 3, test on holdout → save results to `reports/cross_generator/`
- **Robustness test:** JPEG compression (q=70,50,30), downscale (50%,25%), Gaussian blur (σ=1,2) → save line chart to `reports/robustness/`

---

## Hyperparameter Tuning

| Symptom                         | Try                                                                         |
| ------------------------------- | --------------------------------------------------------------------------- |
| Overfitting (train << val loss) | Increase Dropout to 0.6, add L2, stronger augmentation                      |
| Underfitting / slow convergence | Unfreeze earlier layers sooner, raise LR 3–5×                               |
| val_AUC stuck < 0.85            | Switch to CosineAnnealingLR or add linear warmup                            |
| Cross-generator drop > 10 pp    | Add JPEG compression augmentation, try MixUp                                |
| Robustness drops at JPEG q=50   | Add `albumentations.ImageCompression(quality_lower=40)` to train transforms |

---

## Code Rules

### PyTorch / timm

- `timm.create_model(..., num_classes=0)` + custom head — never pass `num_classes` directly
- Use `model.default_cfg` for ImageNet normalization stats (never hardcode mean/std)
- `model.train()` / `model.eval()` must be explicit before loops

### Dataset

- `Dataset.__getitem__` returns `(image_tensor, label_tensor)` — label as `torch.float32`
- Augmentation with `albumentations`; `train_transform` and `val_transform` are separate — never augment val/test beyond resize+normalize
- DataLoader: `num_workers=4`, `pin_memory=True` when GPU available

### Imports & Config

- Import order: stdlib → third-party → local `src/`. Blank lines between groups. No wildcard imports.
- All hyperparameters and paths from YAML in `configs/` or CLI args — never hardcoded in `src/`
- Artifact paths reference `artifacts/` and `reports/` relative to project root

### Naming

| Thing       | Convention                 | Example                                 |
| ----------- | -------------------------- | --------------------------------------- |
| Classes     | `PascalCase`               | `FaceDataset`, `EfficientNetClassifier` |
| Functions   | `snake_case`               | `train_one_epoch`, `compute_metrics`    |
| Constants   | `UPPER_SNAKE_CASE`         | `IMAGENET_MEAN`                         |
| Checkpoints | `{model}_{stage}_best.pth` | `efficientnet_b0_stage1_best.pth`       |

### Reproducibility

```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

Log `torch.__version__`, `timm.__version__`, and GPU name at the start of each run.

---

## Safety Rules

- **Do NOT modify `data/raw/`** — it is the immutable source
- **Do NOT delete checkpoints in `artifacts/`** without explicit confirmation
- **Do NOT run `rm -rf`** or any destructive command without confirmation
- **Do NOT edit files in `src/`** unless explicitly asked — read them freely
- Always read a script or config before executing it

---

## Key Output Locations

| Output                  | Path                                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------- |
| Checkpoints             | `artifacts/efficientnet_b0_stage{1,2,3}_best.pth`                                       |
| Grad-CAM overlays       | `artifacts/gradcam/`                                                                    |
| EDA figures             | `reports/class_balance.png`, `reports/sample_grid_*.png`, `reports/pixel_intensity.png` |
| Eval metrics            | `reports/eval_results.json`                                                             |
| Training log            | `reports/train_log.csv` or `artifacts/train_log.csv`                                    |
| Cross-generator results | `reports/cross_generator/`                                                              |
| Robustness chart        | `reports/robustness/`                                                                   |
| Report draft            | `reports/report_draft.md`                                                               |
| LaTeX assets            | `reports/latex/`                                                                        |

---

## Report Structure (15–20 pages)

> **Language:** All report content must be written in **Vietnamese**. Technical terms (model names, metric names, library/package names) may remain in English.

1. Abstract (~250 words)
2. Introduction — problem, motivation, contributions
3. Related Work — cite 7 papers from `task_plan_ai_face_detection.md`
4. Methodology — dataset, preprocessing, model architecture, 3-stage training
5. Experiments & Results — main table (Acc/P/R/F1/AUC), cross-generator table, robustness chart, Grad-CAM observations
6. Discussion — what worked, limitations, future work
7. Conclusion — 3–4 sentences only
8. References

When writing any section: read `reports/` and `artifacts/` for real numbers first. Use `[INSERT: metric]` placeholders if files don't exist yet. Never fabricate values.
