---
name: report-drafting
description: 'Draft or fill sections of the DS200 AI face detection academic report. Use for: write abstract, draft introduction, write related work, fill methodology, write results section, draft discussion, write conclusion, create slide outline, document findings from artifacts/.'
argument-hint: "Which section — e.g. 'Draft the Methodology section', 'Write abstract', 'Create slide outline', 'Fill Results from artifacts/'"
---

# Report Drafting

## When to Use

- Writing any section of the 15–20 page DS200 project report
- Generating a first draft from experiment artifacts in `reports/` or `artifacts/`
- Creating a 12–15 slide deck outline

> **Language:** All drafted content must be in **Vietnamese**. Technical terms (EfficientNet-B0, BCEWithLogitsLoss, AUC-ROC, etc.) stay in English.

## Procedure

### 1. Scan Available Evidence

Before writing anything, search for and read these files in order:

**Metrics and results** (read all that exist):

- `reports/eval_results.json` or `reports/eval_results.txt` — test set metrics (Acc, P, R, F1, AUC)
- `reports/train_log.csv` or `artifacts/train_log.csv` — per-epoch loss/accuracy curves
- `reports/cross_generator/` — cross-generator confusion matrices and accuracy
- `reports/robustness/` — robustness accuracy table or chart data
- Any `*.json`, `*.csv`, or `*.txt` file under `reports/` or `artifacts/` that contains numeric results

**Figures** (note file paths for LaTeX `\includegraphics`):

- `reports/class_balance.png`, `reports/sample_grid_*.png`, `reports/pixel_intensity.png` — dataset EDA
- `artifacts/gradcam/` — Grad-CAM heatmap overlays
- `reports/robustness/` — robustness line charts
- `reports/cross_generator/` — cross-generator confusion matrices

**Project context**:

- `task_plan_ai_face_detection.md` — reference papers and plan
- `README.md` — project overview and team roles

**Artifact reading rules:**

- If a file contains a metrics table or JSON, extract the numbers verbatim — do not round or summarize unless asked
- If `reports/` and `artifacts/` are empty, write placeholder text using `[INSERT: metric name]` tags and tell the user which files to generate first
- If multiple checkpoint files exist in `artifacts/`, use the one with the highest epoch number or `best_` prefix

### 2. Follow the Report Structure

```
1. Abstract          (~250 words)
2. Introduction      (1–2 pages) — problem, motivation, contributions
3. Related Work      (1–2 pages) — deepfake detection survey, key papers
4. Methodology       (3–4 pages) — dataset, preprocessing, model, training stages
5. Experiments       (3–4 pages) — setup, metrics, results tables, figures
6. Discussion        (1–2 pages) — analysis, limitations, failure cases
7. Conclusion        (0.5–1 page) — summary + future work
8. References        — cite from task_plan_ai_face_detection.md
```

### 3. Section-Specific Guidance

**Abstract** — problem → approach (4 datasets, EfficientNet-B0, 3-stage fine-tuning) → key results → conclusion. ≤250 words.

**Introduction** — hook: rise of generative AI / deepfake threat → gap: single-dataset training limits generalization → contributions (bulleted, parallel structure). No results here.

**Related Work** — cite all 7 papers from `task_plan_ai_face_detection.md`. Group by theme (detection methods, datasets, explainability). Each paragraph ends stating how this work differs.

**Methodology** sub-sections:

1. Dataset — 4 sources, label normalization table, counts, 70/15/15 split
2. Preprocessing — resize 224×224, ImageNet normalization, augmentation list (train only)
3. Model — EfficientNet-B0 + `Linear(1280,256)→ReLU→Dropout(0.5)→Linear(256,1)`
4. Training — 3-stage table (layers, LR, max epochs, callbacks)

**Experiments** — read actual metrics from `reports/`. Include: main test set table (Acc, P, R, F1, AUC), cross-generator table, robustness chart description, Grad-CAM observations.

When writing the Results section, always produce this table first, filled from real file values:

```
| Method                  | Acc   | Prec  | Recall | F1    | AUC   |
|-------------------------|-------|-------|--------|-------|-------|
| Stage 1 (head only)     | X.XX  | X.XX  | X.XX   | X.XX  | X.XX  |
| Stage 2 (+blocks 6-7)   | X.XX  | X.XX  | X.XX   | X.XX  | X.XX  |
| Stage 3 (full fine-tune)| X.XX  | X.XX  | X.XX   | X.XX  | X.XX  |
```

If cross-generator results exist, also produce:

```
| Train on        | Test on         | Acc   | AUC   |
|-----------------|-----------------|-------|-------|
| All 4 datasets  | test split      | X.XX  | X.XX  |
| 3 datasets      | held-out source | X.XX  | X.XX  |
```

**Discussion** — what worked, limitations (accuracy drop on unseen generators, JPEG robustness), future work (frequency-domain features, Stable Diffusion data, C2PA, calibration).

**Conclusion** — 3–4 sentences only. No new information.

### 4. Slide Outline (when requested)

12–15 slides: Title → Problem & Motivation → Datasets → Pipeline → Model Architecture → Training Stages → Main Results → Cross-Generator Results → Robustness Results → Grad-CAM Examples → Demo → Discussion & Limitations → Conclusion → References.

### 5. Save Output

Save drafts to `reports/report_draft.md`. End every response with: **"Next section to write: [name]"**

## Constraints

- Never fabricate metric values — use only numbers found in project files or explicitly provided by the user
- Never overwrite existing content without showing a diff first
- Only create files under `reports/`
