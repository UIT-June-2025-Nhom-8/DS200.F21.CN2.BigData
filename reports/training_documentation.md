# Tài Liệu Huấn Luyện Mô Hình — Phát Hiện Ảnh Khuôn Mặt Tạo Bởi AI

> **Môn học:** DS200.F21.CN2 — Trường Đại học Công nghệ Thông tin, VNU-HCM
> **Đề tài:** Phát hiện ảnh khuôn mặt Real/Fake (AI-generated)
> **Ngày hoàn thành training:** 24/05/2026

---

## 1. Tổng Quan

### 1.1 Mục Tiêu

Xây dựng mô hình học sâu phân loại nhị phân ảnh khuôn mặt thành hai lớp:
- **Real (0):** ảnh chụp thật từ người thật
- **Fake (1):** ảnh tạo bởi AI (GAN, StyleGAN, Deepfake)

Mô hình được huấn luyện trên tập dữ liệu tổng hợp từ 4 nguồn Kaggle, sử dụng kiến trúc EfficientNet-B0 với chiến lược fine-tuning 3 giai đoạn (progressive unfreezing).

### 1.2 Pipeline Tổng Thể

```
Raw Data (4 Kaggle datasets)
    → Tiền xử lý (resize 224x224, normalize ImageNet)
    → Data Augmentation (chỉ tập train)
    → 3-Stage Training (progressive unfreezing)
    → Evaluation (test set độc lập)
    → Output: Checkpoints + Metrics + Visualizations
```

### 1.3 Thời Gian Huấn Luyện

- **Tổng thời gian:** ~17 giờ (từ 1:00 đến 18:00)
- **Môi trường:** NVIDIA GeForce RTX 5070, 12.82 GB VRAM
- **Mixed precision (FP16):** Giảm thời gian ~1.5-2× so với FP32

---

## 2. Môi Trường Huấn Luyện

### 2.1 Phần Cứng

| Thành phần       | Thông số                    |
|------------------|----------------------------|
| CPU              | AMD Ryzen 5 7500F 6-Core   |
| GPU              | NVIDIA GeForce RTX 5070    |
| VRAM             | 12.82 GB                   |
| Compute Capability | 12.0 (Blackwell)         |
| CUDA             | 13.0                       |
| cuDNN            | 92000                      |

### 2.2 Phần Mềm

| Thư viện        | Phiên bản      |
|----------------|---------------|
| Python         | 3.12          |
| PyTorch        | 2.12.0+cu130  |
| timm           | 1.0.27        |
| albumentations | 2.0.8         |
| numpy          | 2.4.4         |
| scikit-learn   | 1.8.0         |
| matplotlib     | ≥3.8          |
| pandas         | ≥2.1          |
| Pillow         | ≥10.0         |

### 2.3 Reproducibility

Toàn bộ quá trình được cố định seed để đảm bảo tái lập kết quả:

```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 3. Tập Dữ Liệu

### 3.1 Nguồn Dữ Liệu

Dữ liệu được tổng hợp từ 4 bộ dataset trên Kaggle:

| # | Tên Dataset                  | Loại Fake         | Đặc điểm                  |
|---|------------------------------|-------------------|---------------------------|
| 1 | 140k Real and Fake Faces     | StyleGAN2         | 70k real (Flickr) + 70k fake |
| 2 | Deepfake and Real Images     | Manipulation-based| Ảnh 256×256               |
| 3 | Fake-Vs-Real-Faces (Hard)    | GAN-based         | Khó phân biệt, tập nhỏ    |
| 4 | Real and Fake Face Detection | ciplab GAN        | Nhiều phương pháp sinh    |

### 3.2 Tiền Xử Lý Dữ Liệu

- **Dedup:** Loại bỏ ảnh trùng lặp bằng MD5 hash
- **Chuẩn hóa nhãn:** Tất cả nhãn được map về `Real` / `Fake`
- **Split:** 70% train / 15% val / 15% test, stratified, `random_state=42`

### 3.3 Thống Kê Tập Dữ Liệu

| Tập   | Tổng     | Real    | Fake    | Tỷ lệ Real |
|-------|----------|---------|---------|------------|
| Train | 221,571  | 104,819 | 116,752 | 47.3%      |
| Val   | 47,479   | 22,461  | 25,018  | 47.3%      |
| Test  | 47,480   | 22,461  | 25,019  | 47.3%      |
| **Tổng** | **316,530** | **149,741** | **166,789** | — |

**Nhận xét:** Tập dữ liệu có mất cân bằng lớp nhẹ (class imbalance) (~47% Real vs ~53% Fake). Chênh lệch này đủ nhỏ để không cần điều chỉnh trọng số lớp trong hàm mất mát, nhưng vẫn cần theo dõi recall của lớp Real trong quá trình đánh giá.

---

## 4. Tiền Xử Lý Ảnh Và Augmentation

### 4.1 Pipeline Tiền Xử Lý

- **Resize:** 224×224 pixels — kích thước mặc định của EfficientNet-B0
- **Normalization:** Sử dụng ImageNet mean/std
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Lý do:** Backbone EfficientNet-B0 được pretrained trên ImageNet → cần cùng phân bố chuẩn hóa

### 4.2 Data Augmentation (chỉ tập train)

| Kỹ thuật                  | Tham số                              | Xác suất | Mục đích |
|---------------------------|--------------------------------------|----------|----------|
| HorizontalFlip            | —                                    | 0.5      | Mô phỏng ảnh gương mặt đối hướng |
| RandomBrightnessContrast  | brightness_limit=0.2, contrast_limit=0.2 | 0.5   | Mô phỏng ánh sáng khác nhau |
| ShiftScaleRotate          | shift=0.1, scale=0.1, rotate=15°    | 0.5      | Mô phỏng tư thế/góc máy thay đổi |
| CoarseDropout             | max_holes=8, kích thước 16×16px     | 0.3      | Giảm thiểu hiện tượng học vẹt đặc trưng cục bộ |

**Tập val/test:** Không áp dụng augmentation — chỉ resize 224×224 + normalize để đánh giá trung thực.

---

## 5. Kiến Trúc Mô Hình

### 5.1 Backbone: EfficientNet-B0

- **Nguồn:** `timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)`
- **Tham số:** ~5.3M (nhẹ hơn ResNet-50 ~25M)
- **Lý do chọn:**
  - Accuracy tương đương ResNet-50 nhưng tham số ít hơn ~5×
  - Training nhanh hơn, ít nguy cơ overfitting trên tập dữ liệu có domain hỗn hợp
  - Pretrained trên ImageNet → feature extractor tốt ngay từ đầu

### 5.2 Custom Classification Head

```
Feature Map (B, 1280, 7, 7)
    → AdaptiveAvgPool2d(1)
    → Flatten (B, 1280)
    → Linear(1280 → 256)
    → ReLU
    → Dropout(0.5)
    → Linear(256 → 1)
    → Logit (B, 1)
```

| Thành phần          | Lý do                                                    |
|---------------------|----------------------------------------------------------|
| 1280 → 256          | Đủ rộng để học ranh giới phân loại Real/Fake              |
| Dropout(0.5)        | Chống overfitting — ảnh fake có artifact dễ "học thuộc"   |
| Output: 1 logit     | Binary classification → dùng BCEWithLogitsLoss           |

### 5.3 Loss Function

`BCEWithLogitsLoss` — kết hợp sigmoid + binary cross-entropy trong một hàm, ổn định số học hơn so với tách riêng `sigmoid()` + `BCELoss`.

---

## 6. Chiến Lược Huấn Luyện 3 Giai Đoạn

### 6.1 Progressive Unfreezing

Chiến lược này dựa trên nguyên tắc **discriminative fine-tuning**: các layer gần đầu vào (backbone) đã có feature tốt từ ImageNet nên cần LR thấp; các layer mới (head) cần học từ đầu nên cần LR cao.

| Stage | Tên                | Lớp Trainable          | Learning Rate | Mục đích |
|-------|--------------------|------------------------|---------------|----------|
| 1     | Freeze Backbone    | Head only              | 1e-4          | Học ranh giới phân loại cơ bản |
| 2     | Unfreeze Top       | Head + blocks 5-6      | 1e-5          | Adapt high-level features |
| 3     | Unfreeze All       | Toàn bộ mô hình        | 1e-6          | Fine-tuning toàn diện |

### 6.2 Hyperparameters

| Tham số           | Giá trị        | Lý do |
|-------------------|---------------|-------|
| Batch size        | 128           | Fit 12GB VRAM với mixed precision |
| Optimizer         | AdamW         | Weight decay tách rời → tốt hơn cho fine-tuning |
| Weight decay      | 1e-4          | Giá trị mặc định từ paper AdamW |
| Max epochs/stage  | 30            | Đủ để hội tụ, EarlyStopping sẽ dừng sớm nếu cần |
| Mixed precision   | FP16 (autocast + GradScaler) | Tăng tốc ~1.5-2×, giảm VRAM |
| DataLoader workers| 2             | Cân bằng tốc độ load data vs RAM usage |

### 6.3 Callbacks

| Callback              | Tham số                      | Tác dụng |
|-----------------------|------------------------------|----------|
| EarlyStopping         | patience=5, monitor=val_loss | Dừng khi model không cải thiện sau 5 epoch |
| ReduceLROnPlateau     | patience=3, factor=0.3, min_lr=1e-7 | Giảm LR khi val_loss bão hòa |
| ModelCheckpoint       | monitor=val_loss, save_top_k=1 | Lưu checkpoint tốt nhất mỗi stage |

### 6.4 Quy Trình Thực Thi

Mô hình được huấn luyện tuần tự qua 3 stage trong một lần chạy `scripts/run_train.py`:

1. **Stage 1:** Head được khởi tạo random, backbone frozen → train đến epoch 30
2. Load checkpoint tốt nhất của Stage 1 → **Stage 2:** Unfreeze blocks 5-6 → train đến epoch 30
3. Load checkpoint tốt nhất của Stage 2 → **Stage 3:** Unfreeze toàn bộ → train đến epoch 30
4. Load checkpoint tốt nhất của Stage 3 → **Evaluation** trên test set độc lập
5. Lưu metrics, confusion matrix, ROC curve, training curves

### 6.5 Kết Quả Training Theo Stage

| Stage | Epochs chạy | Val Loss  | Val Accuracy | Checkpoint Size |
|-------|-------------|-----------|-------------|-----------------|
| 1     | 30/30       | 0.310262  | 0.8663      | 20.3 MB         |
| 2     | 30/30       | 0.138344  | 0.9454      | 26.0 MB         |
| 3     | 30/30       | 0.069906  | 0.9734      | 52.5 MB         |

**Quan sát quan trọng:**
- Cả 3 stage đều chạy đủ 30 epochs — EarlyStopping **không** kích hoạt
- Điều này cho thấy mô hình vẫn tiếp tục cải thiện đến epoch cuối ở mỗi stage
- Val loss giảm đều: 0.310 → 0.138 → 0.070 (giảm ~77% từ Stage 1 sang Stage 3)
- Kích thước checkpoint tăng dần do số lượng layer được lưu optimizer state tăng lên

---

## 7. Kết Quả Đánh Giá

### 7.1 Test Set Metrics

Đánh giá trên tập test độc lập (47,480 ảnh) với checkpoint tốt nhất của Stage 3:

| Metric    | Giá trị | Diễn giải |
|-----------|---------|-----------|
| Accuracy  | 0.9733  | 97.33% ảnh được phân loại đúng |
| Precision | 0.9773  | Trong số ảnh mô hình dự đoán là Fake, 97.73% thực sự là Fake |
| Recall    | 0.9718  | Mô hình phát hiện được 97.18% ảnh Fake thực tế |
| F1-Score  | 0.9745  | Trung bình điều hòa của Precision và Recall |
| AUC-ROC   | 0.9970  | Gần như hoàn hảo trong việc phân biệt Real vs Fake |

### 7.2 Phân Tích Kết Quả

**Điểm mạnh:**
- AUC-ROC = 0.9970 → mô hình có khả năng phân tách Real/Fake rất tốt
- Precision cao (0.9773) cho thấy khi mô hình dự đoán một ảnh là Fake, xác suất đúng rất cao
- Recall cao (0.9718) cho thấy mô hình phát hiện được phần lớn ảnh Fake thực tế
- Kết quả cải thiện rõ rệt qua từng stage: 86.6% → 94.5% → 97.3%

**Hạn chế cần lưu ý:**
- Tập dữ liệu có mất cân bằng lớp nhẹ (53% Fake). Tuy nhiên, Precision cao (0.9773) bác bỏ giả thuyết mô hình thiên vị về lớp đa số — vì nếu mô hình đoán thiên về Fake, số False Positive sẽ tăng và Precision giảm. Kết quả cho thấy mô hình thực sự học được cách phân loại chính xác chứ không phải lợi dụng chênh lệch số lượng mẫu
- Chưa kiểm thử cross-generator — chưa biết mô hình tổng quát hóa tốt đến đâu trên dữ liệu từ nguồn chưa từng thấy
- Chưa kiểm thử robustness với nhiễu ảnh (JPEG compression, blur, downscale)

### 7.3 Visualizations

Các hình ảnh được lưu trong `artifacts/results/`:

| File                    | Nội dung |
|-------------------------|----------|
| `confusion_matrix.png`  | Ma trận nhầm lẫn trên test set |
| `roc_curve.png`         | Đường cong ROC với AUC |
| `training_curves.png`   | Loss và Accuracy curves của 3 stages |
| `metrics.txt`           | Bảng metrics dạng text |

---

## 8. Cấu Trúc Mã Nguồn

### 8.1 File Chính

| File                    | Vai trò |
|-------------------------|---------|
| `configs/train_config.yaml` | Toàn bộ hyperparameters, có inline rationale |
| `scripts/run_train.py`  | Entry point — chạy full pipeline 3 stages |
| `scripts/run_evaluate.py` | Đánh giá checkpoint riêng lẻ |
| `scripts/test_setup.py` | Kiểm tra môi trường trước khi train |
| `src/model.py`          | Định nghĩa FaceDetectionModel |
| `src/dataset.py`        | FaceDataset + create_dataloaders |
| `src/trainer.py`        | Trainer class — training loop + callbacks |
| `src/evaluate.py`       | Evaluator class — metrics + visualizations |

### 8.2 Output Files

```
artifacts/
├── checkpoints/
│   ├── best_stage1.pth   # Head-only (20.3 MB, epoch 30, val_loss=0.3103)
│   ├── best_stage2.pth   # Head + blocks 5-6 (26.0 MB, epoch 30, val_loss=0.1383)
│   └── best_stage3.pth   # Full model (52.5 MB, epoch 30, val_loss=0.0699) ← FINAL
└── results/
    ├── metrics.txt           # Test set metrics
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── training_curves.png
```

**Checkpoint format:**
```python
{
    "epoch": int,               # Số epoch đã chạy
    "model_state_dict": dict,   # weights
    "optimizer_state_dict": dict,
    "val_loss": float,          # Validation loss tốt nhất
    "val_acc": float,           # Validation accuracy tương ứng
    "config": dict              # Cấu hình tại thời điểm train
}
```

---

## 9. Hướng Dẫn Chạy Lại

### 9.1 Cài Đặt Môi Trường

```bash
# Tạo virtual environment (Python 3.12)
python -m venv venv312

# Windows
venv312\Scripts\activate

# Linux/macOS
source venv312/Scripts/activate

# Cài PyTorch với CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Cài dependencies còn lại
pip install -r requirements.txt
```

### 9.2 Chuẩn Bị Dữ Liệu

```bash
# Chạy notebooks theo thứ tự (nếu data/splits/ trống)
# 1. notebooks/00_download_datasets.ipynb — tải 4 dataset từ Kaggle
# 2. notebooks/01_dataset_preparation.ipynb — dedup, split, EDA
```

### 9.3 Chạy Training

```bash
# Kiểm tra setup trước
python scripts/test_setup.py

# Chạy full 3-stage training + evaluation
python scripts/run_train.py

# Hoặc đánh giá checkpoint riêng
python scripts/run_evaluate.py --checkpoint artifacts/checkpoints/best_stage3.pth
```

### 9.4 Điều Chỉnh Hyperparameters

Mọi hyperparameters nằm trong `configs/train_config.yaml`. Một số tuning phổ biến:

| Vấn đề                     | Điều chỉnh |
|----------------------------|-----------|
| Overfitting (train << val) | Tăng Dropout 0.5→0.6, tăng weight_decay |
| Underfitting               | Tăng LR 3-5×, unfreeze sớm hơn |
| Out of Memory              | Giảm batch_size 128→64, tắt mixed precision |
| Training chậm              | Tăng num_workers (nếu RAM đủ) |

---

## 10. Tài Liệu Tham Khảo

### 10.1 Paper Liên Quan

| Paper | Venue | Link |
|-------|-------|------|
| CIFAKE | IEEE Access, 2024 | https://arxiv.org/abs/2303.14126 |
| GenImage | NeurIPS, 2023 | https://arxiv.org/abs/2306.08571 |
| AIDE | ICLR, 2025 | https://github.com/shilinyan99/AIDE |
| AI-Face | CVPR, 2025 | https://arxiv.org/abs/2406.00783 |
| FaceForensics++ | ICCV, 2019 | https://arxiv.org/abs/1901.08971 |
| Celeb-DF | CVPR, 2020 | https://arxiv.org/abs/1909.12962 |

### 10.2 Thư Viện

- **timm** (PyTorch Image Models): https://github.com/huggingface/pytorch-image-models
- **albumentations**: https://albumentations.ai/
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- **AdamW**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019

---

> **Tóm tắt:** Mô hình EfficientNet-B0 với 3-stage progressive fine-tuning đạt **97.33% accuracy** và **0.9970 AUC-ROC** trên tập test 47,480 ảnh. Training mất ~17 giờ trên RTX 5070 với mixed precision. Cả 3 stage đều chạy đủ 30 epochs (EarlyStopping không kích hoạt), cho thấy chiến lược unfreezing dần dần hiệu quả và mô hình vẫn tiếp tục học đến epoch cuối.
