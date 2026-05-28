# Kế Hoạch Viết Báo Cáo — DS200.F21.CN2 AI Face Detection

> **Mục đích:** Tài liệu này phác thảo chi tiết kế hoạch viết báo cáo chính thức (`reports/report_draft.md`) theo đúng cấu trúc template BaoCao.docx.
> **Người dùng:** Agent PaperWriter hoặc thành viên nhóm thực hiện viết báo cáo.
> **Ngôn ngữ đầu ra:** Tất cả nội dung báo cáo phải bằng **tiếng Việt**. Thuật ngữ kỹ thuật (tên mô hình, tên metric, tên thư viện) giữ nguyên tiếng Anh.

---

## Tổng Quan Cấu Trúc Template (BaoCao.docx)

Template gốc có **7 chương chính** + Tài liệu tham khảo:

| Chương | Tiêu đề | Số mục con |
|--------|---------|-----------|
| 1 | Giới Thiệu Đề Tài | 4 mục |
| 2 | Nghiên Cứu Liên Quan | 4 mục |
| 3 | Bộ Dữ Liệu | 3 mục (EDA sâu) |
| 4 | Phương Pháp Đề Xuất | 2 mục (mô hình + công nghệ) |
| 5 | Thực Nghiệm | 5 mục (pipeline + kết quả + demo) |
| 6 | Kết Luận | 2 mục |
| 7 | Hướng Phát Triển | — |
| — | Tài Liệu Tham Khảo | ≥7 bài báo |

---

## CHƯƠNG 1 — GIỚI THIỆU ĐỀ TÀI

### 1.1 Tính Cấp Thiết Của Đề Tài

**Nội dung cần viết:**
- Bối cảnh: sự bùng nổ của AI sinh ảnh (GAN, Diffusion Models) dẫn đến khủng hoảng niềm tin vào hình ảnh số.
- Hệ quả thực tế: deepfake trong chính trị, lừa đảo danh tính, thao túng truyền thông.
- Thách thức kỹ thuật: khuôn mặt AI-generated ngày càng photorealistic, vượt qua phán đoán thủ công của con người.
- Nhu cầu cấp bách: công cụ phân loại tự động, tin cậy, có khả năng khái quát hóa (cross-generator).

**Độ dài mục tiêu:** 3–4 đoạn.  
**Không cần số liệu thực nghiệm** — dùng lập luận và trích dẫn.

---

### 1.2 Mô Tả Bài Toán

**Nội dung cần viết:**
- **Input:** Ảnh khuôn mặt đơn lẻ (RGB, kích thước bất kỳ sau resize).
- **Output:** Nhãn nhị phân — `0 = fake` (AI-generated), `1 = real`.
- Định nghĩa chính thức bài toán binary classification.
- Ràng buộc thực tế: mô hình phải hoạt động với ảnh từ nhiều nguồn (StyleGAN, deepfake, GAN tổng quát, ciplab).
- Metric đánh giá: Accuracy, Precision, Recall, F1-Score, AUC-ROC.

**Nguồn tham khảo nội bộ:** `CLAUDE.md` → phần Model và Evaluation.

---

### 1.3 Đối Tượng và Phạm Vi Nghiên Cứu

**Nội dung cần viết:**

*Đối tượng nghiên cứu:*
- Ảnh khuôn mặt người thật và do AI tổng hợp.
- Kiến trúc EfficientNet-B0 trong bài toán phân loại ảnh nhị phân.
- Kỹ thuật fine-tuning 3 giai đoạn (staged fine-tuning).
- Kỹ thuật tăng cường dữ liệu (data augmentation) với albumentations.

*Phạm vi nghiên cứu:*
- **Dữ liệu:** 4 bộ Kaggle dưới `data/raw/` (140k-StyleGAN, Deepfake-Real, Hard-FakeReal, ciplab). Tổng ~140k+ ảnh.
- **Phương pháp:** Transfer learning từ ImageNet pretrained, huấn luyện 3 giai đoạn, mixed-precision training.
- **Đánh giá:** Test set chuẩn + cross-generator test + robustness test (JPEG compression, downscale, Gaussian blur).
- **Không trong phạm vi:** Video deepfake, audio deepfake, real-time detection.

---

### 1.4 Mục Tiêu Đề Tài

**Nội dung cần viết — danh sách mục tiêu cụ thể:**
1. Thu thập và hợp nhất 4 bộ dữ liệu → tạo `data/splits/train.csv`, `val.csv`, `test.csv` (70/15/15, stratified).
2. Thực hiện EDA toàn diện: phân bố lớp, phân bố pixel, mẫu đại diện từng nguồn.
3. Xây dựng pipeline tiền xử lý: dedup MD5, resize, normalize, augmentation tách biệt train/val.
4. Huấn luyện EfficientNet-B0 theo 3 giai đoạn (head-only → unfreeze blocks 6–7 → full fine-tune).
5. Đánh giá trên test set: bảng Acc/P/R/F1/AUC; confusion matrix; ROC curve.
6. Cross-generator test: huấn luyện trên 3 bộ, test trên bộ còn lại.
7. Robustness test: JPEG q=70/50/30, downscale 50%/25%, Gaussian blur σ=1/2.
8. Trực quan hóa Grad-CAM để giải thích quyết định của mô hình.

---

## CHƯƠNG 2 — NGHIÊN CỨU LIÊN QUAN

### 2.1 Các Nghiên Cứu Về Phát Hiện Khuôn Mặt Giả Mạo (Face Forgery Detection)

**Nội dung cần viết:**
- Tổng quan các hướng tiếp cận: phân tích tần số (FFT artifacts), phân tích texture, học đặc trưng bằng CNN sâu.
- Đề cập các phương pháp dựa trên binary classifier truyền thống → hạn chế.
- Nêu xu hướng transfer learning từ pretrained ImageNet models.

**Bài báo cần trích dẫn từ `task_plan_ai_face_detection.md`:** lấy ít nhất 2–3 bài liên quan chủ đề này.

---

### 2.2 Các Nghiên Cứu Về GAN và Diffusion-Based Image Synthesis

**Nội dung cần viết:**
- StyleGAN, StyleGAN2 (Karras et al.): tạo khuôn mặt photorealistic → thách thức phát hiện.
- Diffusion models (DDPM, Stable Diffusion): thế hệ mới, artifact khác GAN → domain shift.
- Vì sao mô hình huấn luyện trên GAN có thể fail với diffusion images.

**Bài báo:** ít nhất 2 bài về GAN/diffusion từ `task_plan_ai_face_detection.md`.

---

### 2.3 Các Nghiên Cứu Về EfficientNet và Transfer Learning

**Nội dung cần viết:**
- EfficientNet (Tan & Le, 2019): compound scaling, hiệu quả parameter vs accuracy.
- Lý do chọn EfficientNet-B0: nhẹ, pretrained ImageNet, phù hợp fine-tuning.
- Các ứng dụng transfer learning trong face forgery detection.
- So sánh với ResNet, VGG, ViT (tham khảo ngắn).

**Bài báo:** bài gốc EfficientNet + ít nhất 1 bài ứng dụng trong forgery detection.

---

### 2.4 Điểm Mới Của Đề Tài

**Nội dung cần viết — so sánh với các nghiên cứu trước:**
- Sử dụng bộ dữ liệu đa nguồn (4 Kaggle datasets, 140k+ ảnh), đa dạng hơn các nghiên cứu thường chỉ dùng 1 bộ.
- Chiến lược huấn luyện 3 giai đoạn có kiểm soát chặt (frozen backbone → partial unfreeze → full fine-tune) với EarlyStopping và ReduceLROnPlateau.
- Đánh giá cross-generator (hold-out một bộ dữ liệu) để đo khả năng khái quát hóa thực tế.
- Đánh giá robustness với nhiễu nén JPEG, downscale, Gaussian blur — thực tế hơn benchmark tiêu chuẩn.
- Trực quan hóa Grad-CAM để giải thích mô hình — tăng tính minh bạch.

---

## CHƯƠNG 3 — BỘ DỮ LIỆU

### 3.1 Giới Thiệu và Thu Thập

**Nội dung cần viết:**

*Nguồn dữ liệu — bảng 4 bộ Kaggle:*

| Bộ dữ liệu | Đường dẫn (`data/raw/`) | Quy mô | Nguồn gốc |
|-----------|------------------------|--------|-----------|
| 140k-StyleGAN | `140k-real-and-fake-faces/real_vs_fake/real-vs-fake` | 70k real + 70k fake | StyleGAN2 |
| Deepfake-Real | `deepfake-and-real-images/Dataset` | ~ảnh 256×256 | Manipulation-based |
| Hard-FakeReal | `hardfakevsrealfaces` | Small, hard cases | Mixed |
| ciplab | `real-and-fake-face-detection/real_and_fake_face` | Mixed sources | GAN tổng hợp |

*Quy trình hợp nhất:*
- Suy ra nhãn từ tên thư mục: `real`, `Real`, `training_real` → label 1; `fake`, `Fake`, `training_fake` → label 0.
- Dedup bằng MD5 hash.
- Kiểm tra phân bố lớp (xem `reports/class_balance.png`).
- Phân chia stratified 70/15/15, `random_state=42` → lưu vào `data/splits/`.
- Schema CSV: `image_path` (absolute path, string), `label` (int: 0=fake, 1=real).

*Thống kê kết quả hợp nhất:* [INSERT: tổng số ảnh train/val/test, tỷ lệ real:fake sau split — đọc từ `data/splits/train.csv` khi có]

---

### 3.2 Quy Trình Khai Phá Dữ Liệu (EDA)

> Nguồn: `reports/eda/eda_report.md` và các figure tại `reports/`.

#### 3.2.1 Thống Kê Mô Tả

**Nội dung cần viết:**
- Tổng số ảnh, phân bố theo nhãn (bảng).
- Phân bố theo bộ dữ liệu nguồn.
- Kích thước ảnh: min/max/mean width×height trước resize.
- Số kênh màu (RGB vs Grayscale nếu có).

**Figure cần chèn:** `reports/class_balance.png`

---

#### 3.2.2 Phân Tích Thuộc Tính Hình Ảnh (Pixel Statistics)

**Nội dung cần viết:**
- Phân bố cường độ pixel: real vs fake (mean, std theo kênh R/G/B).
- Histogram pixel intensity — nhận xét sự khác biệt giữa real và fake.
- Entropy ảnh: ảnh fake có thể có entropy thấp hơn do artifact.

**Figure cần chèn:** `reports/pixel_intensity.png`

---

#### 3.2.3 Phân Tích Mẫu Đại Diện (Sample Grid)

**Nội dung cần viết:**
- Lưới ảnh mẫu từng lớp và từng bộ dữ liệu.
- Quan sát trực quan: artifact điển hình của ảnh fake (vùng tai, nền, tóc, mắt).
- Mô tả những đặc điểm mà mắt người khó phân biệt.

**Figure cần chèn:** `reports/sample_grid.png`, `reports/sample_grid_fake.png`, `reports/sample_grid_real.png`

---

#### 3.2.4 Phân Tích Phân Bố Theo Nguồn Dữ Liệu

**Nội dung cần viết:**
- So sánh phân bố pixel/màu sắc giữa 4 bộ dữ liệu → domain diversity.
- Nhận xét khó khăn khi huấn luyện cross-domain.
- Lý do cần cross-generator evaluation.

---

#### 3.2.5 Phân Tích Đa Biến — Tương Quan

**Nội dung cần viết:**
- Tương quan giữa pixel stats và nhãn (real/fake).
- Tương quan giữa nguồn dữ liệu và độ khó phân loại (Hard-FakeReal vs 140k-StyleGAN).
- Kết luận về đặc điểm dữ liệu ảnh hưởng đến thiết kế pipeline.

---

### 3.3 Nhận Xét Chất Lượng Bộ Dữ Liệu

**Nội dung cần viết theo các khía cạnh:**

*Kích thước và cấu trúc:*
- Tổng số ảnh sau dedup: [INSERT]
- Cân bằng lớp: [INSERT] — nhận xét cần dùng pos_weight hay không.

*Chất lượng ảnh:*
- Các trường hợp ảnh bị lỗi (corrupt, kích thước quá nhỏ) nếu có.
- Tỷ lệ ảnh bị trùng lặp đã loại bỏ.

*Thách thức kỹ thuật:*
- Domain shift giữa 4 bộ dữ liệu (resolution, color space, GAN type).
- Bộ `hardfakevsrealfaces`: đặc biệt khó, nhận xét riêng.
- Rủi ro data leakage nếu không dedup kỹ.

---

## CHƯƠNG 4 — PHƯƠNG PHÁP ĐỀ XUẤT

### 4.1 Mô Hình

#### 4.1.1 EfficientNet-B0 Backbone

**Nội dung cần viết:**
- Kiến trúc compound scaling của EfficientNet (Tan & Le, 2019): scale đồng thời depth/width/resolution.
- EfficientNet-B0: baseline nhỏ nhất, ~5.3M parameters, input 224×224.
- Tại sao chọn B0: cân bằng accuracy vs inference speed, phù hợp pretrain ImageNet.
- Tạo model: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)` → feature extractor (output 1280-dim).

*Sơ đồ kiến trúc (tùy chọn TikZ):* Backbone → Global Average Pooling → Custom Head.

---

#### 4.1.2 Custom Classification Head

**Nội dung cần viết — mô tả kiến trúc head:**

```
Linear(1280, 256) → ReLU → Dropout(0.5) → Linear(256, 1)
```

- Lý do dùng BCEWithLogitsLoss: kết hợp sigmoid + BCE, ổn định số học hơn BCE(sigmoid(x)).
- Không áp dụng sigmoid trước loss — chỉ áp dụng khi inference.
- Xử lý class imbalance: `pos_weight` nếu cần.

---

#### 4.1.3 Chiến Lược Huấn Luyện 3 Giai Đoạn

**Bảng tóm tắt 3 giai đoạn (dạng bảng như BaoCao.docx):**

| Giai đoạn | Lớp được huấn luyện | Learning Rate | Epochs | Điều kiện dừng |
|-----------|--------------------|-----------|----|----------------|
| 1 | Head only (backbone frozen) | 1e-4 | 20–30 | EarlyStopping patience=5 |
| 2 | Head + EfficientNet blocks 6–7 | 1e-5 | 20–30 | EarlyStopping patience=5 |
| 3 | Full model | 1e-6 | 20–30 | EarlyStopping + ReduceLROnPlateau |

**Nội dung giải thích thêm:**
- Lý do freeze backbone ở Stage 1: tránh phá vỡ pretrained features khi head chưa hội tụ.
- Lý do unfreeze dần ở Stage 2–3: progressive fine-tuning giảm catastrophic forgetting.
- Mixed-precision training (AMP): `torch.cuda.amp.autocast()` + `GradScaler` — tăng tốc 1.5–2× trên GPU.
- Batch size 128; checkpoint lưu theo `val_loss` tốt nhất mỗi giai đoạn.

---

#### 4.1.4 Data Augmentation

**Nội dung cần viết:**

*Train transform (albumentations):*
- RandomResizedCrop, HorizontalFlip, ColorJitter, RandomBrightnessContrast
- Gaussian noise, JPEG compression augmentation (quality_lower=40 nếu robustness kém)
- Normalize với ImageNet mean/std từ `model.default_cfg`

*Val/Test transform:*
- Resize → CenterCrop → Normalize (không augment thêm)

---

### 4.2 Công Nghệ Sử Dụng

**Nội dung cần viết — bảng công nghệ:**

| Loại | Công nghệ | Mục đích |
|------|-----------|---------|
| Framework | PyTorch | Deep learning |
| Model library | timm | EfficientNet-B0 pretrained |
| Augmentation | albumentations | Train transform |
| Metrics | scikit-learn | Acc/P/R/F1/AUC-ROC |
| Visualization | matplotlib | Confusion matrix, ROC, Grad-CAM |
| Môi trường | Google Colab / Local GPU | Training |
| Config | YAML (`configs/`) | Hyperparameter management |

---

## CHƯƠNG 5 — THỰC NGHIỆM

### 5.1 Quy Trình Thực Nghiệm

#### 5.1.1 Xử Lý Dữ Liệu

**Nội dung cần viết — từng bước cụ thể:**

- **Bước 1:** Download 4 bộ Kaggle (`notebooks/00_download_datasets.ipynb`). Xác thực tính toàn vẹn.
- **Bước 2:** MD5 dedup — loại bỏ ảnh trùng lặp giữa các bộ.
- **Bước 3:** Kiểm tra class balance per dataset. Vẽ `reports/class_balance.png`.
- **Bước 4:** EDA: pixel intensity, sample grid, thống kê kích thước ảnh.
- **Bước 5:** Stratified split 70/15/15 → `data/splits/train.csv`, `val.csv`, `test.csv`.
- **Bước 6:** Lưu EDA figures vào `reports/`.

---

#### 5.1.2 Chuẩn Bị Dữ Liệu Cho Mô Hình

**Nội dung cần viết:**
- `FaceDataset` class: đọc CSV, load ảnh, apply transform.
- `DataLoader`: batch_size=128, num_workers=4, pin_memory=True (khi có GPU).
- Reproducibility seeds: random=42, numpy=42, torch=42.
- Log phiên bản: `torch.__version__`, `timm.__version__`, tên GPU.

---

### 5.2 Độ Đo Đánh Giá

**Nội dung cần viết — mô tả từng metric (format giống BaoCao: công thức + ý nghĩa + ưu/nhược):**

1. **Accuracy** — tỷ lệ phân loại đúng tổng thể.
2. **Precision** — trong số những ảnh được dự đoán là fake, bao nhiêu thực sự là fake.
3. **Recall (Sensitivity)** — trong số ảnh fake thực, bao nhiêu được phát hiện đúng.
4. **F1-Score** — harmonic mean của Precision và Recall.
5. **AUC-ROC** — diện tích dưới đường ROC; đo khả năng phân biệt ở mọi ngưỡng threshold.

*Đặc biệt cho bài toán này:* Recall quan trọng hơn Precision vì bỏ sót ảnh fake nguy hiểm hơn false alarm.

---

### 5.3 Thông Số Chi Tiết Các Giai Đoạn Huấn Luyện

**Nội dung cần viết — bảng hyperparameter từng stage:**

*Stage 1 — Head Only:*
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- EarlyStopping: patience=5, monitor=val_loss
- Checkpoint: `artifacts/efficientnet_b0_stage1_best.pth`

*Stage 2 — Partial Unfreeze:*
- Unfreeze EfficientNet blocks 6–7
- Optimizer: AdamW, lr=1e-5
- Load từ Stage 1 checkpoint

*Stage 3 — Full Fine-tune:*
- Unfreeze toàn bộ model
- Optimizer: AdamW, lr=1e-6
- EarlyStopping + ReduceLROnPlateau kết hợp

*[INSERT: training log table từ `reports/train_log.csv` hoặc `artifacts/train_log.csv` khi có]*

---

### 5.4 Đánh Giá Thực Nghiệm

#### 5.4.1 Kết Quả Thực Nghiệm

**Bảng kết quả chính — test set:**

| Mô hình | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------|---------|-----------|--------|---------|---------|
| EfficientNet-B0 (Stage 1) | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| EfficientNet-B0 (Stage 2) | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| EfficientNet-B0 (Stage 3) | [INSERT] | [INSERT] | [INSERT] | [INSERT] | [INSERT] |

> Đọc từ `reports/eval_results.json` khi có. Dùng `[INSERT: metric]` nếu chưa có.

**Figure cần chèn:**
- Confusion matrix: `reports/confusion_matrix.png`
- ROC curve: `reports/roc_curve.png`

---

**Bảng Cross-Generator:**

| Holdout Dataset | Train trên | Accuracy | F1 | AUC |
|----------------|-----------|---------|-----|-----|
| 140k-StyleGAN | 3 bộ còn lại | [INSERT] | [INSERT] | [INSERT] |
| Deepfake-Real | 3 bộ còn lại | [INSERT] | [INSERT] | [INSERT] |
| Hard-FakeReal | 3 bộ còn lại | [INSERT] | [INSERT] | [INSERT] |
| ciplab | 3 bộ còn lại | [INSERT] | [INSERT] | [INSERT] |

> Đọc từ `reports/cross_generator/` khi có.

---

**Bảng Robustness:**

| Perturbation | Acc | F1 | AUC |
|-------------|-----|-----|-----|
| Baseline (no noise) | [INSERT] | [INSERT] | [INSERT] |
| JPEG q=70 | [INSERT] | [INSERT] | [INSERT] |
| JPEG q=50 | [INSERT] | [INSERT] | [INSERT] |
| JPEG q=30 | [INSERT] | [INSERT] | [INSERT] |
| Downscale 50% | [INSERT] | [INSERT] | [INSERT] |
| Downscale 25% | [INSERT] | [INSERT] | [INSERT] |
| Gaussian blur σ=1 | [INSERT] | [INSERT] | [INSERT] |
| Gaussian blur σ=2 | [INSERT] | [INSERT] | [INSERT] |

> Figure: `reports/robustness/robustness_chart.png` (line chart)

---

#### 5.4.2 Nhận Xét

**Nội dung cần viết theo 4 tiểu mục (theo format BaoCao.docx):**

*5.4.2.1 Đánh giá tổng quan:*
- Mô hình tốt nhất stage nào, metric nào nổi bật.
- So sánh stage 1 vs 2 vs 3: improvement qua từng bước fine-tuning.

*5.4.2.2 Đánh giá theo từng độ đo:*
- AUC-ROC: phân tích ở mọi threshold.
- Recall vs Precision tradeoff.
- F1 trên Hard-FakeReal: đặc biệt khó.

*5.4.2.3 Phân tích nguyên nhân:*
- Nếu cross-generator drop > 10 pp: domain shift, thiếu diversity augmentation.
- Nếu robustness giảm ở JPEG q=30: thiếu compression augmentation trong train.
- Nếu val_AUC stuck < 0.85: cần CosineAnnealingLR hoặc warmup.

*5.4.2.4 Grad-CAM Observations:*
- Mô hình tập trung vào vùng nào: tai, nền, vùng tiếp giáp tóc-da — artifact phổ biến của GAN.
- Figure: `artifacts/gradcam/` — chọn ví dụ điển hình real vs fake.

---

### 5.5 Ứng Dụng Thực Nghiệm (Demo)

**Nội dung cần viết:**
- Mô tả pipeline inference: load ảnh → transform → forward pass → sigmoid → threshold 0.5.
- Demo script: `scripts/demo.py` (nếu đã implement).
- Giao diện minh họa (nếu có web demo): chèn screenshots.
- Grad-CAM overlay: hình ảnh minh họa vùng model tập trung.

---

## CHƯƠNG 6 — KẾT LUẬN

### 6.1 Tóm Tắt Đề Tài

**Nội dung cần viết (~3–4 câu):**
- Đề tài đặt ra bài toán gì (binary classification real vs AI-generated faces).
- Phương pháp tiếp cận: EfficientNet-B0, 3-stage fine-tuning, 4 Kaggle datasets.
- Kết quả đạt được: [INSERT: metric tốt nhất] trên test set.
- Đóng góp đánh giá: cross-generator + robustness + Grad-CAM.

---

### 6.2 Đóng Góp Của Đề Tài

**Nội dung cần viết — danh sách đóng góp:**
1. Xây dựng pipeline hợp nhất 4 bộ dữ liệu đa nguồn (140k+ ảnh) với dedup và stratified split.
2. Áp dụng chiến lược fine-tuning 3 giai đoạn có kiểm soát cho bài toán forgery detection.
3. Đánh giá cross-generator toàn diện — đo khả năng tổng quát hóa thực tế.
4. Đánh giá robustness với nhiều loại nhiễu thực tế (JPEG, blur, downscale).
5. Trực quan hóa Grad-CAM giải thích quyết định mô hình, hỗ trợ tin cậy hóa AI.

---

## CHƯƠNG 7 — HƯỚNG PHÁT TRIỂN ĐỀ TÀI

**Nội dung cần viết:**
1. **Mở rộng kiến trúc:** thử EfficientNet-B4, EfficientNet-B7, hoặc Vision Transformer (ViT) để so sánh.
2. **Tập dữ liệu mới:** bổ sung ảnh từ Stable Diffusion, Midjourney — các model sinh ảnh mới nhất.
3. **Detection video deepfake:** mở rộng pipeline sang phân tích frame-by-frame hoặc temporal models.
4. **Real-time inference:** tối ưu model (quantization, pruning) để deploy trên thiết bị biên.
5. **Multi-task learning:** kết hợp phân loại real/fake với localization artifact (segmentation).
6. **Adversarial robustness:** đánh giá với adversarial examples (FGSM, PGD) — quan trọng trong bối cảnh tấn công có chủ đích.

---

## TÀI LIỆU THAM KHẢO

**Danh sách bài báo cần trích dẫn (≥7 bài — lấy từ `task_plan_ai_face_detection.md`):**

> Đọc `task_plan_ai_face_detection.md` để lấy đầy đủ danh sách. Dưới đây là placeholder theo chủ đề:

1. [EfficientNet] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. [StyleGAN] Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for GANs. CVPR.
3. [Face Forgery Detection] — [INSERT từ task_plan]
4. [Transfer Learning for Forgery] — [INSERT từ task_plan]
5. [Deepfake Detection Survey] — [INSERT từ task_plan]
6. [Grad-CAM] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
7. [albumentations] Buslaev, A., et al. (2020). Albumentations: Fast and Flexible Image Augmentations. Information.
8. [timm] Wightman, R. (2019). PyTorch Image Models. GitHub.

---

## Hướng Dẫn Thực Thi Kế Hoạch

### Thứ Tự Viết Các Mục (Khuyến Nghị)

```
Bước 1: Chương 3 (Bộ Dữ Liệu) — đọc data splits và EDA figures trước
Bước 2: Chương 4 (Phương Pháp) — không cần kết quả thực nghiệm
Bước 3: Chương 5 (Thực Nghiệm) — cần artifacts/ và reports/results/
Bước 4: Chương 2 (Nghiên Cứu Liên Quan) — cần task_plan_ai_face_detection.md
Bước 5: Chương 1 (Giới Thiệu) — viết sau khi hiểu rõ toàn bộ đề tài
Bước 6: Chương 6 + 7 (Kết Luận + Hướng Phát Triển)
Bước 7: Abstract — viết sau cùng
```

### Nguồn Dữ Liệu Cho Từng Phần

| Phần báo cáo | Đọc file nào |
|-------------|-------------|
| Số liệu EDA | `reports/eda/eda_report.md`, `reports/class_balance.png`, `reports/pixel_intensity.png` |
| Training details | `reports/training_documentation.md`, `reports/train_log.csv` |
| Kết quả test | `reports/eval_results.json`, `reports/confusion_matrix.png`, `reports/roc_curve.png` |
| Cross-generator | `reports/cross_generator/` |
| Robustness | `reports/robustness/` |
| Grad-CAM | `artifacts/gradcam/` |
| Checkpoints info | `artifacts/checkpoints/` |
| Reference papers | `task_plan_ai_face_detection.md` |

### Quy Tắc Bắt Buộc (từ CLAUDE.md)

- Không bịa số liệu. Dùng `[INSERT: metric]` nếu file chưa có.
- Tất cả văn bản báo cáo bằng **tiếng Việt**. Giải thích thuật ngữ kỹ thuật khi lần đầu xuất hiện.
- Chương 3 (EDA): ngôn ngữ đơn giản, tránh jargon, giải thích bằng phép so sánh.
- Chương 1, 2, 4, 5: ngôn ngữ học thuật trang trọng.
- Lưu bản thảo tại: `reports/report_draft.md`.
- Lưu LaTeX tại: `reports/latex/`.
