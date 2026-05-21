---
description: 'Generate exploratory data analysis code for the AI face detection datasets — class distributions, per-source breakdowns, sample grids, pixel statistics, duplicate visualization'
tools: [read, search, edit]
---

> Read `notebooks/01_dataset_preparation.ipynb` for the existing baseline EDA (class balance, sample grids, pixel intensity). Then generate **additional** analysis cells as requested below.

Generate a self-contained Jupyter notebook cell (or a sequence of cells) for the specified EDA task.

## Available Data

After running `notebooks/01_dataset_preparation.ipynb`, the following files exist:

| File                           | Content                                            |
| ------------------------------ | -------------------------------------------------- |
| `data/splits/train.csv`        | columns: `image_path`, `label` (0=fake, 1=real)    |
| `data/splits/val.csv`          | same schema                                        |
| `data/splits/test.csv`         | same schema                                        |
| `reports/class_balance.png`    | Already generated — real/fake bar chart            |
| `reports/sample_grid_real.png` | Already generated — random real image grid         |
| `reports/sample_grid_fake.png` | Already generated — random fake image grid         |
| `reports/pixel_intensity.png`  | Already generated — mean pixel intensity histogram |

The 4 source datasets live under `data/raw/`:

- `140k-real-and-fake-faces/real_vs_fake/real-vs-fake` → label from subdir (`real`/`fake`)
- `deepfake-and-real-images/Dataset` → label from subdir (`Real`/`Fake`)
- `hardfakevsrealfaces` → label from subdir (`real`/`fake`)
- `real-and-fake-face-detection/real_and_fake_face` → label from subdir (`training_real`/`training_fake`)

## EDA Tasks (generate whichever is requested)

### A. Per-source class distribution

Bar chart with one group per dataset showing real/fake counts. Useful for spotting which dataset dominates.

```python
# Expected: grouped bar chart, one group per dataset name inferred from image_path
# Save to: reports/eda_per_source_distribution.png
```

### B. Image resolution distribution

Scatter or 2D histogram of (width × height) for a sample of images. Flag images that deviate from 224×224 expectation.

```python
# Sample up to 2000 images from train.csv
# Save to: reports/eda_resolution_scatter.png
```

### C. Aspect ratio analysis

Histogram of width/height ratios. Confirm all images are square (ratio ≈ 1.0).

```python
# Save to: reports/eda_aspect_ratio.png
```

### D. Mean pixel intensity per source

Box plot of mean pixel values per dataset to detect brightness/contrast domain shifts between generators.

```python
# Save to: reports/eda_intensity_per_source.png
```

### E. Duplicate sample viewer

If MD5 duplicates were found during dedup, display a grid of example duplicate pairs (before dedup) to understand what was removed.

```python
# Requires storing hash→paths mapping during dedup; generate cells to re-compute if not already stored
```

### F. Train / val / test overlap check

Verify that `image_path` values in train, val, and test sets are fully disjoint.

```python
# Should print: "No overlap detected" or show any overlapping paths
```

### G. Label noise probe (confidence sampling)

Load the best available checkpoint from `artifacts/`, run inference on 50 random training images, and display the 10 images with lowest confidence (hardest for the model to classify).

```python
# Requires checkpoint; skip if artifacts/ is empty
# Save grid to: reports/eda_low_confidence_samples.png
```

## Code Requirements

- Follow `.github/instructions/code-style.instructions.md`
- All figures: `figsize=(12, 5)` or wider; `dpi=150`; `tight_layout()`; saved with `plt.savefig(...)`; `plt.close()` after saving
- **Figure titles, axis labels, and legend text must be in Vietnamese.** Metric names and technical terms (e.g., `Real`, `Fake`, `AUC`) may stay in English.
- Use `pathlib.Path` for all file paths
- Each cell should be independently runnable (import what it needs at the top)
- Print a one-line summary after each cell in Vietnamese (e.g., `"Đã lưu: reports/eda_per_source_distribution.png"`)
