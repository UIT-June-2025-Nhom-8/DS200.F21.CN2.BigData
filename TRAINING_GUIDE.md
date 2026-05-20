# Training Guide - AI-Generated Face Detection

## Tổng quan

Training pipeline gồm 3 stages (progressive unfreezing):
- **Stage 1**: Freeze backbone, train head only (LR=1e-4)
- **Stage 2**: Unfreeze top 2 blocks (LR=1e-5)
- **Stage 3**: Unfreeze all (LR=1e-6)

Ước tính thời gian: **3-5 giờ** trên RTX 4070/5070 12GB VRAM

---

## Cài đặt

### 1. Cài dependencies

```bash
pip install -r requirements.txt
```

### 2. Kiểm tra GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## Chạy Training

### Cách 1: Chạy script trực tiếp

```bash
cd scripts
python run_train.py
```

Script sẽ tự động:
1. Load config từ `configs/train_config.yaml`
2. Load datasets từ `data/splits/*.csv`
3. Train 3 stages liên tiếp
4. Save checkpoints vào `artifacts/checkpoints/`
5. Evaluate trên test set
6. Save metrics + plots vào `artifacts/results/`

### Cách 2: Chạy từ root directory

```bash
python scripts/run_train.py
```

---

## Cấu trúc Output

Sau khi train xong, bạn sẽ có:

```
artifacts/
├── checkpoints/
│   ├── best_stage1.pth    # Best model từ Stage 1
│   ├── best_stage2.pth    # Best model từ Stage 2
│   └── best_stage3.pth    # Best model từ Stage 3 (final)
└── results/
    ├── metrics.txt        # Test set metrics (accuracy, F1, AUC)
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_curves.png  # Loss/accuracy curves cho 3 stages
```

---

## Hyperparameters & Rationale

Tất cả hyperparameters đã được config sẵn trong `configs/train_config.yaml`. Dưới đây là tóm tắt:

### Data
- **Image size**: 224×224 (EfficientNet-B0 default)
- **Batch size**: 128 (fit 12GB VRAM với mixed precision)
- **Normalize**: ImageNet mean/std (model pretrained trên ImageNet)
- **Augmentation** (train only):
  - HorizontalFlip (p=0.5)
  - RandomBrightnessContrast (p=0.5)
  - ShiftScaleRotate (p=0.5)
  - CoarseDropout (p=0.3)

### Model
- **Backbone**: EfficientNet-B0 pretrained (timm)
  - Nhẹ hơn ResNet (5.3M vs 25M params)
  - Accuracy tương đương ResNet-50
- **Head**: Linear(1280→256) → ReLU → Dropout(0.5) → Linear(256→1)
  - 256 hidden dim: đủ rộng cho decision boundary
  - Dropout 0.5: chống overfitting

### Training
- **Mixed precision**: Bật (FP16) → nhanh ~1.5-2× + tiết kiệm VRAM
- **Loss**: BCEWithLogitsLoss (standard cho binary classification)
- **Optimizer**: AdamW (weight_decay=1e-4)
  - Tốt hơn Adam cho fine-tuning
  - weight_decay=1e-4 từ paper AdamW (Loshchilov & Hutter, 2019)

### Learning Rates (3 stages)
- **Stage 1**: 1e-4 (head mới → cần LR cao)
- **Stage 2**: 1e-5 (fine-tune top blocks → LR thấp hơn 10×)
- **Stage 3**: 1e-6 (fine-tune toàn bộ → LR rất thấp)

### Callbacks
- **EarlyStopping**: patience=5 (dừng nếu val_loss không giảm sau 5 epochs)
- **ReduceLROnPlateau**: patience=3, factor=0.3 (giảm LR xuống 30% nếu plateau)
- **ModelCheckpoint**: save best model theo val_loss

### Max Epochs
- Mỗi stage: tối đa 30 epochs
- EarlyStopping thường dừng sớm (~10-15 epochs cho Stage 1)

---

## Nguồn gốc Hyperparameters

| Param | Giá trị | Nguồn |
|-------|---------|-------|
| Image size 224 | 224×224 | EfficientNet-B0 default (timm docs) |
| Batch size | 128 | Fit 12GB VRAM với mixed precision |
| Normalize | ImageNet mean/std | Model pretrained trên ImageNet |
| Backbone | EfficientNet-B0 | Nhẹ hơn ResNet, accuracy tương đương |
| LR stages | 1e-4 → 1e-5 → 1e-6 | Discriminative fine-tuning (best practice) |
| AdamW weight_decay | 1e-4 | Default từ paper AdamW |
| Dropout | 0.5 | Chống overfitting với fake artifacts |
| EarlyStopping patience | 5 | Model thường overfit sau ~5 epochs không cải thiện |
| ReduceLR patience | 3 | Giảm LR khi gần optimum |

---

## Tuning (nếu cần)

Nếu kết quả chưa đạt yêu cầu, có thể tune:

1. **Learning rate**: Tăng/giảm LR từng stage trong `train_config.yaml`
2. **Augmentation**: Tăng/giảm strength (brightness_limit, rotate_limit...)
3. **Dropout**: Tăng nếu overfit, giảm nếu underfit
4. **Batch size**: Giảm nếu out of memory, tăng nếu còn VRAM

---

## Troubleshooting

### Out of Memory (OOM)
- Giảm `batch_size` trong config (128 → 64 → 32)
- Hoặc tắt mixed precision (set `mixed_precision: false`)

### Training quá chậm
- Kiểm tra GPU đang được dùng: `nvidia-smi`
- Giảm `num_workers` trong config (4 → 2 → 0)

### Val loss không giảm
- Kiểm tra data augmentation có quá mạnh không
- Thử giảm learning rate
- Kiểm tra class imbalance (có thể cần class weights)

---

## Sau khi Training

1. **Kiểm tra metrics** trong `artifacts/results/metrics.txt`
2. **Xem training curves** trong `artifacts/results/training_curves.png`
3. **Phân tích confusion matrix** để hiểu lỗi model
4. **Tiếp tục với các bước tiếp theo** trong task plan:
   - Cross-Generator Testing (Section 6)
   - Robustness Testing (Section 7)
   - Grad-CAM Explainability (Section 8)
   - Demo Webapp (Section 9)

---

## Liên hệ

Nếu gặp vấn đề, tham khảo:
- Task plan: `task_plan_ai_face_detection.md`
- Config file: `configs/train_config.yaml`
- Notebook tham khảo: https://www.kaggle.com/code/sohailasayedmohamed/deepfake-face-detection-resnet-acc-97
