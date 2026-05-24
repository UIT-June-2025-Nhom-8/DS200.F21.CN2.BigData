# Quick Start - Training AI-Generated Face Detection

## Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Bước 2: Kiểm tra setup

```bash
python scripts/test_setup.py
```

Script này sẽ verify:
- Config file load được
- CUDA/GPU available
- Data loading hoạt động
- Model tạo được
- Forward pass chạy được

## Bước 3: Chạy training

```bash
python scripts/run_train.py
```

Training sẽ chạy 3 stages tự động (~3-5 giờ trên RTX 4070/5070).

## Bước 4: Xem kết quả

Sau khi train xong, check:
- **Metrics**: `artifacts/results/metrics.txt`
- **Plots**: `artifacts/results/*.png`
- **Best model**: `artifacts/checkpoints/best_stage3.pth`

---

## Files đã tạo

### Code
- `src/dataset.py` - Dataset loader với augmentation
- `src/model.py` - EfficientNet-B0 + custom head
- `src/trainer.py` - 3-stage training engine
- `src/evaluate.py` - Metrics + visualization

### Scripts
- `scripts/run_train.py` - Main training script
- `scripts/run_evaluate.py` - Standalone evaluation
- `scripts/test_setup.py` - Verify setup trước khi train

### Config & Docs
- `configs/train_config.yaml` - Tất cả hyperparameters + rationale
- `TRAINING_GUIDE.md` - Hướng dẫn chi tiết
- `HYPERPARAMETERS.md` - Bảng tóm tắt params + nguồn gốc
- `requirements.txt` - Python dependencies

---

## Hyperparameters Chính

| Param | Value | Lý do |
|-------|-------|-------|
| Model | EfficientNet-B0 | Nhẹ (5.3M params), accuracy cao |
| Batch size | 128 | Fit 12GB VRAM với mixed precision |
| LR Stage 1 | 1e-4 | Train head mới |
| LR Stage 2 | 1e-5 | Fine-tune top blocks |
| LR Stage 3 | 1e-6 | Fine-tune toàn bộ |
| Optimizer | AdamW (wd=1e-4) | Tốt cho fine-tuning |
| Loss | BCEWithLogitsLoss | Standard binary classification |
| Mixed precision | Enabled | Nhanh ~1.5-2×, tiết kiệm VRAM |

Chi tiết đầy đủ: xem `HYPERPARAMETERS.md`

---

## Troubleshooting

**Out of Memory?**
→ Giảm `batch_size` trong `configs/train_config.yaml` (128 → 64)

**Training quá chậm?**
→ Check GPU: `nvidia-smi`

**Val loss không giảm?**
→ Thử giảm learning rate hoặc tăng augmentation

---

## Tiếp theo

Sau khi train xong baseline:
1. Cross-Generator Testing (task plan section 6)
2. Robustness Testing (section 7)
3. Grad-CAM Explainability (section 8)
4. Demo Webapp (section 9)

Xem chi tiết trong `task_plan_ai_face_detection.md`
