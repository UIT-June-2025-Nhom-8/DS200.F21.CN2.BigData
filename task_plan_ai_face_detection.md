# Detect AI-Generated Faces — Task Plan

**Đề tài:** Phát hiện ảnh khuôn mặt tạo bởi AI | **4 tuần, 5 người**
**Stack:** PyTorch + timm | **GPU:** RTX 4070/5070 12GB VRAM (local)

---

## 1. Dataset

4 dataset (tổng ~10 GB):

| # | Dataset | Mô tả | Link |
|---|---------|-------|------|
| 1 | 140k Real and Fake Faces | 70k real (Flickr) + 70k fake (StyleGAN) | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces |
| 2 | Deepfake and Real Images | Ảnh 256×256, manipulated faces | https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images |
| 3 | Fake-Vs-Real-Faces (Hard) | Bộ nhỏ, ảnh khó phân biệt | https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces |
| 4 | Real and Fake Face Detection | Dataset từ ciplab | https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection |

**Lưu ý:** Notebook gốc chỉ dùng 1 dataset để train. Nhóm gộp cả 4 bộ để tăng đa dạng data (StyleGAN, manipulation-based, GAN-based, ciplab).

**Notebook tham khảo:**
- https://www.kaggle.com/code/sohailasayedmohamed/deepfake-face-detection-resnet-acc-97

---

## 2. Tiền xử lý

- Gộp 4 dataset thành 1 file CSV chung (cột: image_path, label). Chuẩn hóa label thành "Real" / "Fake" vì mỗi bộ đặt tên khác nhau (real, Real, training_real, 0, 1...)
- Resize toàn bộ về **224×224**
- Split **70/15/15** (train/val/test) — notebook gốc chỉ split 80/20 train/val, không có test set riêng → nhóm cần tách test set riêng để đánh giá chính xác hơn
- Normalize chuẩn ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] — notebook gốc chỉ rescale 1/255, chưa tối ưu cho pretrained model
- Augmentation cho train set: HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, CoarseDropout — notebook gốc không có augmentation
- Val/test set: chỉ resize + normalize, KHÔNG augment
- Kiểm tra ảnh corrupt, duplicate cross-dataset trước khi train (hash-based dedup)
- Kiểm tra class imbalance sau khi gộp, dùng `class_weight` nếu tỉ lệ real/fake lệch > 60/40

---

## 3. EDA

- Visualize sample ảnh real vs fake từ mỗi dataset (grid so sánh)
- Thống kê số lượng ảnh mỗi dataset, tỉ lệ real/fake tổng
- Histogram pixel intensity real vs fake
- FFT spectrum real vs fake (phát hiện GAN artifacts trong frequency domain)

---

## 4. Model

**Backbone đã chốt: EfficientNet-B0** (timm: `efficientnet_b0`, pretrained ImageNet)

```python
import timm
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
```

- Classification head: `model.classifier` → `Linear(1280, 256) → ReLU → Dropout(0.5) → Linear(256, 1)`
- **Batch size: 128** | **Mixed precision: bật** (`torch.cuda.amp`) — tận dụng 12GB VRAM
- Ước tính train toàn bộ 3 stage: **3-5 tiếng** trên RTX 4070/5070
- Training 3 giai đoạn:

| Giai đoạn | Layers trainable | LR | Mô tả |
|-----------|-----------------|-----|-------|
| Stage 1 | Freeze toàn bộ backbone | 1e-4 | Chỉ train head, cho model học phân biệt real/fake cơ bản |
| Stage 2 | Unfreeze top 2 MBConv blocks (blocks 6-7) | 1e-5 | Fine-tune high-level features của backbone |
| Stage 3 | Unfreeze toàn bộ | 1e-6 | Fine-tune toàn bộ model, kết hợp ReduceLROnPlateau |

- Loss: Binary Cross-Entropy with Logits. Optimizer: AdamW (weight_decay=1e-4)
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (patience=3, factor=0.3), ModelCheckpoint (save best theo val_loss)
- Mỗi stage train tối đa 20-30 epoch, EarlyStopping sẽ dừng sớm nếu không cải thiện

---

## 5. Evaluation

- Tính: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Vẽ: Confusion Matrix, ROC Curve, training/validation loss & accuracy curve
- Đánh giá trên **test set riêng** (không phải validation set như notebook gốc)

---

## 6. Kiểm thử 1 — Cross-Generator

Mục đích: model train trên mix nhiều loại fake → test trên loại fake chưa thấy.

- Tách 1 trong 4 dataset ra làm hold-out test (ví dụ: tách bộ 140k StyleGAN ra, chỉ train trên 3 bộ còn lại)
- Test model trên bộ hold-out → đánh giá khả năng generalize sang loại fake khác
- Hoặc cách khác: tách theo loại generator — train trên manipulation-based, test trên GAN-based
- Ghi nhận accuracy drop, vẽ confusion matrix riêng

---

## 7. Kiểm thử 2 — Robustness

Mục đích: đánh giá model dưới điều kiện ảnh bị biến đổi thực tế.

- Các perturbation cần test:
  - JPEG compression: quality 70, 50, 30
  - Resize xuống rồi phóng lại: 50%, 25%
  - Gaussian blur: σ=1.0, σ=2.0
- Mỗi perturbation apply lên toàn bộ test set → tính accuracy
- Vẽ line chart: mức perturbation vs accuracy drop

---

## 8. Explainability (Grad-CAM)

- Dùng thư viện `pytorch-grad-cam` (`pip install grad-cam`)
- Chạy Grad-CAM trên best model, visualize heatmap overlay lên ~10 ảnh real + 10 ảnh fake
- Phân tích: model nhìn vào đâu? (mắt, tóc, nền, tai, viền mặt?)

---

## 9. Demo Webapp

- Dùng Gradio
- Chức năng: upload ảnh → hiển thị Real/Fake + confidence % + Grad-CAM heatmap
- Quay video demo backup phòng trường hợp live demo fail khi trình bày

---

## 10. Báo cáo & Slide

**Báo cáo (15-20 trang):** Abstract → Introduction → Related Work → Methodology → Experiments & Results → Discussion → Conclusion & Future Work

**Lưu ý trong Discussion:** thừa nhận limitations — model phụ thuộc vào loại fake trong training data, robustness drop dưới compression, confidence chưa calibrated. Đề xuất hướng cải thiện: frequency-domain features, thêm data từ generator mới, C2PA provenance.

---

## Tài liệu tham khảo gợi ý

1. Bird & Lotfi, 2024. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* IEEE Access — https://arxiv.org/abs/2303.14126
2. Zhu et al., 2023. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image.* NeurIPS 2023 — https://arxiv.org/abs/2306.08571
3. Yan et al., 2025. *AIDE: A Sanity Check for AI-generated Image Detection.* ICLR 2025 — https://github.com/shilinyan99/AIDE
4. Lin et al., 2025. *AI-Face: AI-Generated Face Dataset and Fairness Benchmark.* CVPR 2025 — https://arxiv.org/abs/2406.00783
5. Rössler et al., 2019. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019 — https://arxiv.org/abs/1901.08971
6. Li et al., 2020. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics.* CVPR 2020 — https://arxiv.org/abs/1909.12962
7. Roy et al., 2026. *A Comprehensive Dataset for Human vs. AI Generated Image Detection.* — https://arxiv.org/abs/2601.00553
