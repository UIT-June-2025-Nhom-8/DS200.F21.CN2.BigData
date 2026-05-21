---
name: code-review
description: 'Review Python source files for correctness, style, and safety. Use for: review code, check src/, audit scripts, is this correct, validate dataset.py, check model.py, review train.py, review eval.py, check imports, CLAUDE.md compliance, PyTorch best practices, security check, generate patch, apply fix.'
argument-hint: "What to review — e.g. 'Review src/model.py', 'Audit all src/ files', 'Check scripts/prepare_data.py before running', 'Review everything'"
---

# Code Review

## When to Use

- Before running any new or recently modified script in `src/` or `scripts/`
- When the user asks "review", "check my code", "is this correct?", "validate", "audit", or "apply fix"
- After writing a new module to catch issues before the first training run

## Scope

Review all files matching these paths unless the user specifies a subset:

| Path pattern        | What to check                    |
| ------------------- | -------------------------------- |
| `src/*.py`          | Full checklist (A through G)     |
| `scripts/*.py`      | Full checklist (A through G)     |
| `configs/*.yaml`    | Category B (config hygiene) only |
| `notebooks/*.ipynb` | Code cells only, full checklist  |

## Procedure

### Step 1 — Discover Files

```bash
ls src/ scripts/ configs/ notebooks/ 2>/dev/null
```

List all files in scope. Read each file fully before checking — never check from memory alone.

### Step 2 — Apply Review Checklist

#### Category A — PyTorch / timm Correctness

| ID  | Rule                                                                                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------------- |
| A1  | `timm.create_model(...)` uses `num_classes=0` + custom head — never `num_classes=1` or any non-zero value passed directly |
| A2  | Normalization stats come from `model.default_cfg` — never hardcode `mean=[0.485, 0.456, 0.406]`                           |
| A3  | `model.train()` called before training loop; `model.eval()` called before val/test loop — both explicit                   |
| A4  | No `torch.sigmoid()` applied to logits before passing to `BCEWithLogitsLoss` — the loss already includes sigmoid          |
| A5  | Loss function is `BCEWithLogitsLoss`, not `CrossEntropyLoss` or `BCELoss`                                                 |
| A6  | Device line: `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — never `"cuda:0"` hardcoded                  |
| A7  | Labels in `Dataset.__getitem__` returned as `torch.float32` (not `int` or `long`)                                         |
| A8  | `DataLoader` uses `num_workers=4` and `pin_memory=True` when GPU is available                                             |
| A9  | Mixed precision uses `torch.cuda.amp.autocast()` + `GradScaler`, not manual casting                                       |

#### Category B — Config Hygiene

| ID  | Rule                                                                                                          |
| --- | ------------------------------------------------------------------------------------------------------------- |
| B1  | No literal learning rates in `src/` or `scripts/` (e.g. `lr = 1e-4`) — must read from YAML config or CLI args |
| B2  | No hardcoded batch sizes in `src/` or `scripts/`                                                              |
| B3  | No hardcoded absolute file paths (e.g. `"/Users/..."`, `"/home/..."`) — use config or project-relative paths  |
| B4  | No hardcoded architectural hyperparameters (`dropout=0.5`, `hidden=256`) in src/ — from config only           |

#### Category C — Dataset and Augmentation

| ID  | Rule                                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------ |
| C1  | `train_transform` and `val_transform` are separate objects — `val_transform` must not include random augmentations |
| C2  | `val_transform` (and `test_transform`) contains **only** resize and normalize — no flips, crops, color jitter      |
| C3  | `Dataset.__getitem__` returns a `(image_tensor, label_tensor)` tuple — label is a scalar float32 tensor            |

#### Category D — Import Order

| ID  | Rule                                                                                   |
| --- | -------------------------------------------------------------------------------------- |
| D1  | Import order: stdlib → third-party → local `src/` with a blank line between each group |
| D2  | No wildcard imports (`from module import *`)                                           |

#### Category E — Naming Conventions

| ID  | Rule                                                                                            |
| --- | ----------------------------------------------------------------------------------------------- |
| E1  | Class names: `PascalCase` (e.g. `FaceDataset`, `EfficientNetClassifier`)                        |
| E2  | Function and variable names: `snake_case` (e.g. `train_one_epoch`, `compute_metrics`)           |
| E3  | Module-level constants: `UPPER_SNAKE_CASE` (e.g. `IMAGENET_MEAN`)                               |
| E4  | Checkpoint filenames follow `{model}_{stage}_best.pth` (e.g. `efficientnet_b0_stage1_best.pth`) |

#### Category F — Reproducibility

| ID  | Rule                                                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- |
| F1  | Seed block present at run start: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)` |
| F2  | Version logging at run start: `torch.__version__`, `timm.__version__`, and GPU name (via `torch.cuda.get_device_name(0)`)           |

#### Category G — Safety and Security

| ID  | Rule                                                                                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------------- |
| G1  | No hardcoded credentials, API keys, passwords, or tokens anywhere in the file                                             |
| G2  | No `eval()` or `exec()` called on user-supplied strings or data loaded from external files                                |
| G3  | No `rm -rf`, `shutil.rmtree`, or destructive shell commands without an explicit guard and user confirmation               |
| G4  | No `subprocess.run(shell=True, input=<external_data>)` — always use argument lists, never shell=True with untrusted input |

### Step 3 — Produce Output

Format findings as the report below. If no issues are found in a severity category, omit that section entirely.

## Output Format

````
## Kết quả Code Review: <filepath>

### Lỗi cần sửa trước khi chạy

- **[A4] Dòng 42:** `logits = torch.sigmoid(output)` được truyền vào loss — loại bỏ sigmoid, `BCEWithLogitsLoss` đã tích hợp sẵn.
- **[B3] Dòng 8:** `DATA_DIR = "/home/user/data/splits"` — thay bằng đường dẫn tương đối hoặc đọc từ config.

### Cảnh báo nên sửa

- **[B1] Dòng 15:** `lr = 1e-4` hardcoded — chuyển sang đọc từ YAML config hoặc CLI argument.
- **[C1] Dòng 78:** `val_transform` có chứa `RandomHorizontalFlip` — augmentation không được dùng trên tập validation.

### Gợi ý cải thiện

- **[F2] Dòng 1:** Chưa log `torch.__version__` và tên GPU khi khởi chạy script.

---

### Patch

```diff
--- a/src/train.py
+++ b/src/train.py
@@ -5,7 +5,7 @@
 import os
-DATA_DIR = "/home/user/data/splits"
+DATA_DIR = config["data"]["splits_dir"]

@@ -39,7 +39,6 @@
         outputs = model(inputs)
-        outputs = torch.sigmoid(outputs)
         loss = criterion(outputs, labels)
````

```

Rules for the patch block:
- Include only Errors and Warnings in the patch — never Notes.
- Each hunk uses unified diff format (`--- a/`, `+++ b/`, `@@ ... @@`).
- Keep context lines (unchanged lines) at ±3 around each change.
- If a file has no Errors or Warnings, do not include a patch block for it.
- If reviewing multiple files, produce one patch block per file.

## Applying the Patch

If the user says "apply the patch", "apply the fix", or "fix this", use the file edit tools to apply changes directly to the file. Before applying, confirm with the user if more than 10 lines across all hunks are being changed.

## Constraints

- Read the full file before reporting any issue — never check from memory alone.
- Do not execute any code during review — this phase is read-only.
- Do not modify files unless explicitly asked to apply a patch.
- Do not fabricate line numbers — read the actual file content first.
- Report only issues that are provably present — do not speculate about runtime behavior.
- Do not report an issue that is already handled correctly (e.g. a sigmoid that is not being passed to the loss).
```
