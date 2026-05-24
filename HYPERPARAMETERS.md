# Hyperparameters Summary - AI-Generated Face Detection

**Model**: EfficientNet-B0 (timm pretrained)  
**Hardware**: RTX 4070/5070 12GB VRAM  
**Training Time**: ~3-5 hours (3 stages)  
**Date**: 2026-05-19

---

## 1. Data Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Image Size** | 224×224 | EfficientNet-B0 default input size (from timm documentation) |
| **Batch Size** | 128 | Fits 12GB VRAM with mixed precision (FP16). Without mixed precision: only ~64-96 |
| **Normalization** | mean=[0.485, 0.456, 0.406]<br>std=[0.229, 0.224, 0.225] | ImageNet normalization (model pretrained on ImageNet) |
| **Train/Val/Test Split** | 70% / 15% / 15% | 221,571 / 47,479 / 47,480 images |
| **Num Workers** | 4 | DataLoader parallel workers |

---

## 2. Augmentation (Train Set Only)

| Augmentation | Parameters | Probability | Rationale |
|--------------|------------|-------------|-----------|
| **HorizontalFlip** | - | 0.5 | Faces can be mirrored |
| **RandomBrightnessContrast** | brightness_limit=0.2<br>contrast_limit=0.2 | 0.5 | Lighting variations in real-world images |
| **ShiftScaleRotate** | shift_limit=0.1<br>scale_limit=0.1<br>rotate_limit=15° | 0.5 | Slight pose/angle changes |
| **CoarseDropout** | max_holes=8<br>max_height=16<br>max_width=16 | 0.3 | Prevent overfitting to specific regions (e.g., eyes, mouth) |

**Val/Test sets**: No augmentation (only resize + normalize)

---

## 3. Model Architecture

| Component | Configuration | Rationale |
|-----------|---------------|-----------|
| **Backbone** | EfficientNet-B0 (pretrained) | - Lighter than ResNet (5.3M vs 25M params)<br>- Similar accuracy to ResNet-50<br>- Faster training on limited GPU<br>- Pretrained on ImageNet → good feature extractor |
| **Feature Dim** | 1280 | EfficientNet-B0 output dimension |
| **Classifier Head** | Linear(1280 → 256)<br>→ ReLU<br>→ Dropout(0.5)<br>→ Linear(256 → 1) | - 256: Wide enough for real/fake decision boundary<br>- Dropout 0.5: Prevent overfitting (fake images may have obvious artifacts) |
| **Output** | Single logit | Binary classification with BCEWithLogitsLoss |

---

## 4. Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Loss Function** | BCEWithLogitsLoss | Standard for binary classification, numerically stable |
| **Optimizer** | AdamW | Better than Adam for fine-tuning (decoupled weight decay) |
| **Weight Decay** | 1e-4 | Default from AdamW paper (Loshchilov & Hutter, 2019) |
| **Mixed Precision** | Enabled (FP16) | - RTX 4070/5070 has tensor cores → ~1.5-2× faster<br>- Reduces VRAM usage → allows batch_size=128 |
| **Max Epochs/Stage** | 30 | Stage 1 converges fast (~10-15 epochs), Stage 2-3 need more but EarlyStopping stops early |
| **Random Seed** | 42 | For reproducibility |

---

## 5. Three-Stage Training (Progressive Unfreezing)

**Rationale**: Gradual unfreezing prevents catastrophic forgetting of pretrained features

### Stage 1: Freeze Backbone, Train Head Only

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frozen Layers** | All backbone layers | Preserve ImageNet pretrained features |
| **Trainable Layers** | Classifier head only | Head initialized randomly → needs to learn first |
| **Learning Rate** | 1e-4 | Higher LR for new head to converge quickly |
| **Expected Epochs** | ~10-15 | Head learns basic real/fake separation |

### Stage 2: Unfreeze Top 2 Blocks

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frozen Layers** | Blocks 0-4 | Keep low-level features frozen |
| **Trainable Layers** | Blocks 5-6 + head | Top blocks learn high-level features → adapt to face domain |
| **Learning Rate** | 1e-5 | 10× lower than Stage 1 to avoid destroying pretrained weights |
| **Expected Epochs** | ~15-20 | Fine-tune high-level features |

### Stage 3: Unfreeze All

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frozen Layers** | None | Full fine-tuning |
| **Trainable Layers** | All layers | Maximum adaptation to face detection task |
| **Learning Rate** | 1e-6 | Very low LR to avoid catastrophic forgetting |
| **Expected Epochs** | ~15-25 | Full model adaptation |

---

## 6. Callbacks

| Callback | Configuration | Rationale |
|----------|---------------|-----------|
| **EarlyStopping** | monitor=val_loss<br>patience=5<br>mode=min | Stop if val_loss doesn't improve for 5 epochs → prevent overfitting |
| **ReduceLROnPlateau** | monitor=val_loss<br>patience=3<br>factor=0.3<br>min_lr=1e-7 | If val_loss plateaus for 3 epochs → reduce LR to 30%<br>Helps model fine-tune when near optimum |
| **ModelCheckpoint** | monitor=val_loss<br>mode=min<br>save_top_k=1 | Save best model per stage according to val_loss |

---

## 7. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness |
| **Precision** | Of predicted Fake, how many are actually Fake? |
| **Recall** | Of actual Fake, how many did we detect? |
| **F1-Score** | Harmonic mean of Precision & Recall |
| **AUC-ROC** | Model's ability to distinguish Real vs Fake across all thresholds |
| **Confusion Matrix** | Visualize True Positive, False Positive, True Negative, False Negative |
| **ROC Curve** | Trade-off between True Positive Rate and False Positive Rate |

---

## 8. Sources & References

### Hyperparameter Design Choices

1. **EfficientNet-B0 architecture**: [timm documentation](https://huggingface.co/docs/timm/index)
2. **ImageNet normalization**: Standard preprocessing for pretrained models
3. **Batch size 128**: Empirical testing on 12GB VRAM with mixed precision
4. **3-stage learning rates (1e-4 → 1e-5 → 1e-6)**: Discriminative fine-tuning best practice
   - Howard & Ruder, 2018. *Universal Language Model Fine-tuning for Text Classification.* ACL 2018
5. **AdamW optimizer**: Loshchilov & Hutter, 2019. *Decoupled Weight Decay Regularization.* ICLR 2019
6. **Progressive unfreezing**: Common practice in transfer learning (fastai, timm)
7. **Augmentation choices**: Standard augmentations for face images (Albumentations library)
8. **Dropout 0.5**: Empirical default for preventing overfitting in classifier heads

### Task-Specific Choices

- **Notebook reference**: [Deepfake Face Detection ResNet (Acc 97%)](https://www.kaggle.com/code/sohailasayedmohamed/deepfake-face-detection-resnet-acc-97)
  - Original: ResNet + 80/20 split + rescale 1/255 only
  - Our improvements: EfficientNet-B0 + 70/15/15 split + ImageNet normalize + augmentation + 3-stage training

---

## 9. Expected Results

Based on similar work and notebook reference:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| **Test Accuracy** | 95-98% | Depends on dataset quality and augmentation |
| **Test F1-Score** | 95-98% | Should be close to accuracy if classes balanced |
| **Test AUC-ROC** | 0.98-0.99 | High AUC indicates good separation |
| **Training Time** | 3-5 hours | On RTX 4070/5070 with mixed precision |

---

## 10. Tuning Guidelines (If Needed)

If results don't meet expectations:

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| **Low accuracy (<90%)** | - Underfitting<br>- LR too low | - Increase LR (e.g., 2e-4, 2e-5, 2e-6)<br>- Train more epochs<br>- Reduce dropout |
| **High train acc, low val acc** | Overfitting | - Increase dropout (0.5 → 0.6)<br>- Stronger augmentation<br>- Add weight decay |
| **Val loss not decreasing** | - LR too high<br>- Bad initialization | - Decrease LR<br>- Check data loading (labels correct?) |
| **Out of Memory (OOM)** | Batch size too large | - Reduce batch_size (128 → 64 → 32)<br>- Disable mixed precision |
| **Training too slow** | - CPU bottleneck<br>- Disk I/O | - Reduce num_workers<br>- Check GPU utilization (nvidia-smi) |

---

## 11. File Locations

| File | Path | Description |
|------|------|-------------|
| **Config** | `configs/train_config.yaml` | All hyperparameters in YAML format |
| **Training Script** | `scripts/run_train.py` | Main training entry point |
| **Evaluation Script** | `scripts/run_evaluate.py` | Standalone evaluation script |
| **Dataset Loader** | `src/dataset.py` | PyTorch Dataset + DataLoader |
| **Model** | `src/model.py` | EfficientNet-B0 + custom head |
| **Trainer** | `src/trainer.py` | 3-stage training engine |
| **Evaluator** | `src/evaluate.py` | Metrics + visualization |
| **Checkpoints** | `artifacts/checkpoints/` | Saved model weights |
| **Results** | `artifacts/results/` | Metrics, plots, curves |

---

**Last Updated**: 2026-05-19  
**Author**: Training Team  
**Task Plan**: `task_plan_ai_face_detection.md`
