---
description: "Scaffold the dataset merging, label normalization, dedup, and train/val/test CSV split pipeline for the AI face detection project"
tools: [read, search, edit]
---

Generate a complete `scripts/prepare_data.py` script for the DS200 AI face detection project.

The script must:

## 1. Dataset Root Discovery
Accept a `--data-root` CLI argument pointing to the folder containing the 4 downloaded Kaggle datasets as subfolders. Also accept `--output-dir` (default: `data/splits/`) for CSV output.

## 2. Label Normalization
Scan each dataset subfolder and map all label variants to exactly `Real` or `Fake`:

| Dataset | Raw label variants | Normalized |
|---------|-------------------|-----------|
| 140k Real and Fake Faces | `real`, `fake` (folder names) | `Real` / `Fake` |
| Deepfake and Real Images | `Real`, `Fake` (folder names) | `Real` / `Fake` |
| Fake-Vs-Real-Faces (Hard) | `real`, `fake` (folder names) | `Real` / `Fake` |
| Real and Fake Face Detection | `training_real`, `training_fake`, `validation_real`, `validation_fake` | `Real` / `Fake` |

Build a master DataFrame with columns: `image_path` (absolute), `label` (`Real`/`Fake`), `source_dataset` (dataset name string).

## 3. Corrupt Image Check
Filter out unreadable images using `PIL.Image.open(...).verify()`. Log the count of removed corrupt files.

## 4. Hash-Based Deduplication
Compute MD5 hash of each image file. Remove exact duplicates across all datasets, keeping one copy. Log the count removed.

## 5. Class Imbalance Check
Print and log real/fake counts and ratio after dedup. If ratio exceeds 60/40, print a warning: `"[WARN] Class imbalance detected: consider using class_weight in training."` Do not auto-balance.

## 6. Stratified Train / Val / Test Split
Split 70% / 15% / 15% using `sklearn.model_selection.train_test_split` with `stratify=label` and `random_state=42`. Perform the split twice: first carve out test (15%), then split the remainder into train/val.

## 7. Output
Save three CSV files to `--output-dir`:
- `train.csv`, `val.csv`, `test.csv`
- Columns: `image_path`, `label`, `source_dataset`

Print a final summary table:
```
Split   | Total  | Real   | Fake
--------|--------|--------|------
train   | ...    | ...    | ...
val     | ...    | ...    | ...
test    | ...    | ...    | ...
```

## Code Requirements
- Follow `src/**/*.py` code style (see `.github/instructions/code-style.instructions.md`)
- Use `argparse` for CLI arguments; no hardcoded paths
- Use `pathlib.Path` throughout — no `os.path`
- Add `tqdm` progress bar for the corrupt-check and hash loop
- All log messages use `logging` (not `print`) except the final summary table
- At the top: `random.seed(42); np.random.seed(42)`

Place the final script at `scripts/prepare_data.py`.
