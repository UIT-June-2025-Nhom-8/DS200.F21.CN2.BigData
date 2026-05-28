# Chương 5 — Thực Nghiệm

> **Nguồn số liệu:** `artifacts/results/metrics.txt`, `reports/training_documentation.md`  
> **Hình ảnh:** `artifacts/results/`  
> **Chưa có:** cross-generator (`reports/cross_generator/`), robustness (`reports/robustness/`), Grad-CAM (`artifacts/gradcam/`)  
> **Cập nhật:** 28/05/2026

---

## 5.1 Quy Trình Thực Nghiệm

### 5.1.1 Xử Lý Dữ Liệu

Quy trình chuẩn bị dữ liệu trước huấn luyện được thực hiện qua hai notebook tuần tự:

**Notebook 1 — `notebooks/00_download_datasets.ipynb`:**  
Tải bốn bộ dữ liệu từ Kaggle về thư mục `data/raw/`. Notebook được thiết kế idempotent — bỏ qua các bộ đã tồn tại khi chạy lại. Yêu cầu thông tin xác thực Kaggle được cấu hình sẵn.

**Notebook 2 — `notebooks/01_dataset_preparation.ipynb`:**  
Thực hiện toàn bộ pipeline tiền xử lý theo thứ tự:

1. Đọc toàn bộ ảnh từ bốn nguồn, suy ra nhãn từ tên thư mục (Real=0, Fake=1).
2. Tính MD5 hash từng file để loại bỏ ảnh trùng lặp hoàn toàn.
3. Kiểm tra phân bố lớp theo nguồn, vẽ biểu đồ lưu tại `reports/class_balance.png`.
4. Phân chia stratified 70/15/15 với `random_state=42`.
5. Xuất ba file CSV: `data/splits/train.csv`, `val.csv`, `test.csv`.

Kết quả sau pipeline: **316.530 ảnh** (221.571 train / 47.479 val / 47.480 test), phân bố nhãn đồng đều 47,3% thật — 52,7% giả trên cả ba tập.

### 5.1.2 Chuẩn Bị Dữ Liệu Cho Mô Hình

Dữ liệu được đưa vào mô hình thông qua lớp `FaceDataset` và `DataLoader` (chi tiết tại `src/dataset.py`):

- **Tập huấn luyện:** resize 224×224 → augmentation (HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate, CoarseDropout) → chuẩn hóa ImageNet → `torch.Tensor`.
- **Tập val/test:** resize 224×224 → chuẩn hóa ImageNet → `torch.Tensor` (không augmentation).
- **DataLoader:** batch\_size=128, num\_workers=2, pin\_memory=True.

Seed tái lập được cố định ở mức `42` cho toàn bộ thư viện (Python `random`, NumPy, PyTorch, CUDA). Phiên bản thư viện được ghi nhận tại thời điểm bắt đầu mỗi lần chạy: PyTorch 2.12.0+cu130, timm 1.0.27, GPU NVIDIA GeForce RTX 5070.

---

## 5.2 Độ Đo Đánh Giá

Kết quả cuối cùng được báo cáo **chỉ trên tập test** (47.480 ảnh), không sử dụng tập val để chọn ngưỡng hay điều chỉnh siêu tham số sau huấn luyện. Năm độ đo được tính bằng `sklearn.metrics`:

**Bảng 5.1 — Định nghĩa và ý nghĩa các độ đo đánh giá**

| Độ đo | Công thức | Ý nghĩa trong bài toán này |
|---|---|---|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Tỷ lệ ảnh phân loại đúng tổng thể |
| Precision | TP / (TP+FP) | Trong số ảnh dự đoán là giả, bao nhiêu thực sự là giả |
| Recall | TP / (TP+FN) | Trong số ảnh giả thực tế, bao nhiêu được phát hiện đúng |
| F1-Score | 2·(P·R) / (P+R) | Trung bình điều hòa của Precision và Recall |
| AUC-ROC | Diện tích dưới đường ROC | Khả năng phân biệt Real/Fake ở mọi ngưỡng threshold |

*Lưu ý:* Trong bối cảnh phát hiện ảnh giả mạo, Recall quan trọng hơn Precision — bỏ sót ảnh giả (False Negative) gây hại thực tế nhiều hơn cảnh báo nhầm (False Positive). AUC-ROC là độ đo tổng hợp tin cậy nhất vì không phụ thuộc vào ngưỡng phân loại cụ thể.

*Quy ước nhãn:* Positive class = Fake (1). Precision và Recall được tính theo lớp Fake.

---

## 5.3 Thông Số Chi Tiết Các Giai Đoạn Huấn Luyện

### 5.3.1 Siêu Tham Số Chung

| Tham số | Giá trị | Nguồn |
|---|---|---|
| Optimizer | AdamW | `configs/train_config.yaml` |
| Weight decay | 1×10⁻⁴ | `configs/train_config.yaml` |
| Batch size | 128 | `configs/train_config.yaml` |
| Max epochs/stage | 30 | `configs/train_config.yaml` |
| Mixed precision | FP16 (autocast + GradScaler) | `configs/train_config.yaml` |
| EarlyStopping patience | 5 (monitor: val\_loss) | `configs/train_config.yaml` |
| ReduceLROnPlateau | patience=3, factor=0,3, min\_lr=1×10⁻⁷ | `configs/train_config.yaml` |
| Random seed | 42 | `configs/train_config.yaml` |

### 5.3.2 Thông Số Từng Giai Đoạn

**Giai đoạn 1 — Head only (backbone đóng băng):**
- Learning rate: 1×10⁻⁴
- Lớp huấn luyện: chỉ classification head (~327K tham số)
- Mục đích: Hội tụ nhanh ranh giới phân loại cơ bản trước khi mở băng backbone

**Giai đoạn 2 — Partial unfreeze (blocks 5–6):**
- Learning rate: 1×10⁻⁵
- Lớp huấn luyện: head + EfficientNet blocks 5 và 6 (hai block cuối, học đặc trưng bậc cao)
- Nạp từ checkpoint tốt nhất Giai đoạn 1

**Giai đoạn 3 — Full fine-tune:**
- Learning rate: 1×10⁻⁶
- Lớp huấn luyện: toàn bộ mô hình (~5,3M tham số)
- Nạp từ checkpoint tốt nhất Giai đoạn 2

### 5.3.3 Kết Quả Huấn Luyện Theo Giai Đoạn

**Bảng 5.2 — Kết quả huấn luyện trên tập kiểm định (val set)**

| Giai đoạn | Epochs chạy | Val Loss tốt nhất | Val Accuracy | Checkpoint |
|---|---|---|---|---|
| 1 (Head only) | 30/30 | 0,3103 | 86,63% | `artifacts/checkpoints/best_stage1.pth` (20,3 MB) |
| 2 (Partial unfreeze) | 30/30 | 0,1383 | 94,54% | `artifacts/checkpoints/best_stage2.pth` (26,0 MB) |
| 3 (Full fine-tune) | 30/30 | 0,0699 | 97,34% | `artifacts/checkpoints/best_stage3.pth` (52,5 MB) |

Nguồn: `reports/training_documentation.md`, Mục 6.5.

**Nhận xét:** Cả ba giai đoạn đều chạy hết 30 epochs — EarlyStopping không kích hoạt ở bất kỳ giai đoạn nào. Điều này cho thấy mô hình vẫn tiếp tục cải thiện đến epoch cuối của mỗi giai đoạn; nếu tăng `max_epochs_per_stage` lên cao hơn, kết quả có thể cải thiện thêm. Val loss giảm liên tục 0,310 → 0,138 → 0,070, tương ứng mức cải thiện ~77% từ Giai đoạn 1 đến Giai đoạn 3.

> **Hình 5.1** — Đường cong val\_loss và val\_accuracy theo epoch của 3 giai đoạn  
> Nguồn: `artifacts/results/training_curves.png`

---

## 5.4 Đánh Giá Thực Nghiệm

### 5.4.1 Kết Quả Trên Tập Test

Đánh giá được thực hiện trên tập test độc lập (47.480 ảnh) với checkpoint tốt nhất của Giai đoạn 3 (`best_stage3.pth`, val\_loss=0,0699). Ngưỡng phân loại mặc định là 0,5 trên đầu ra sigmoid.

**Bảng 5.3 — Kết quả đánh giá trên tập test (checkpoint Stage 3)**

| Độ đo | Giá trị |
|---|---|
| Accuracy | **0,9733** (97,33%) |
| Precision | **0,9773** (97,73%) |
| Recall | **0,9718** (97,18%) |
| F1-Score | **0,9745** (97,45%) |
| AUC-ROC | **0,9970** (99,70%) |

Nguồn: `artifacts/results/metrics.txt`.

**So sánh kết quả qua các giai đoạn (trên val set):**

| Giai đoạn | Val Accuracy | Cải thiện so với giai đoạn trước |
|---|---|---|
| 1 (Head only) | 86,63% | — |
| 2 (Partial unfreeze) | 94,54% | +7,91 điểm phần trăm |
| 3 (Full fine-tune) | 97,34% | +2,80 điểm phần trăm |

Mỗi giai đoạn mở băng thêm đều đóng góp cải thiện đáng kể, xác nhận hiệu quả của chiến lược progressive unfreezing. Mức tăng lớn nhất (+7,91 pp) xảy ra khi chuyển từ Giai đoạn 1 sang Giai đoạn 2 — thời điểm các đặc trưng bậc cao của backbone bắt đầu được thích nghi với domain khuôn mặt thật/giả.

> **Hình 5.2** — Ma trận nhầm lẫn (confusion matrix) trên tập test  
> Nguồn: `artifacts/results/confusion_matrix.png`

> **Hình 5.3** — Đường cong ROC với AUC = 0,9970  
> Nguồn: `artifacts/results/roc_curve.png`

---

### 5.4.2 Đánh Giá Cross-Generator

Bài kiểm tra cross-generator đánh giá khả năng tổng quát hóa của mô hình: huấn luyện trên ba bộ dữ liệu, đánh giá trên bộ còn lại chưa từng thấy trong quá trình huấn luyện.

**Bảng 5.4 — Kết quả cross-generator evaluation**

| Bộ dữ liệu holdout | Huấn luyện trên | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|---|
| 140k-StyleGAN | 3 bộ còn lại | [INSERT: cross_gen_acc_stylegan] | [INSERT: cross_gen_f1_stylegan] | [INSERT: cross_gen_auc_stylegan] |
| Deepfake-Real | 3 bộ còn lại | [INSERT: cross_gen_acc_deepfake] | [INSERT: cross_gen_f1_deepfake] | [INSERT: cross_gen_auc_deepfake] |
| Hard-FakeReal | 3 bộ còn lại | [INSERT: cross_gen_acc_hard] | [INSERT: cross_gen_f1_hard] | [INSERT: cross_gen_auc_hard] |
| ciplab | 3 bộ còn lại | [INSERT: cross_gen_acc_ciplab] | [INSERT: cross_gen_f1_ciplab] | [INSERT: cross_gen_auc_ciplab] |

> Điền từ `reports/cross_generator/` khi có kết quả.

---

### 5.4.3 Đánh Giá Robustness

Bài kiểm tra robustness đánh giá độ bền của mô hình trước các biến dạng ảnh thường gặp trong thực tế: nén JPEG (mất thông tin khi chia sẻ qua mạng xã hội), downscale (ảnh độ phân giải thấp), và Gaussian blur (ảnh chụp không nét).

**Bảng 5.5 — Kết quả robustness evaluation**

| Điều kiện nhiễu | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Baseline (không nhiễu) | 0,9733 | 0,9745 | 0,9970 |
| JPEG nén q=70 | [INSERT: rob_acc_jpeg70] | [INSERT: rob_f1_jpeg70] | [INSERT: rob_auc_jpeg70] |
| JPEG nén q=50 | [INSERT: rob_acc_jpeg50] | [INSERT: rob_f1_jpeg50] | [INSERT: rob_auc_jpeg50] |
| JPEG nén q=30 | [INSERT: rob_acc_jpeg30] | [INSERT: rob_f1_jpeg30] | [INSERT: rob_auc_jpeg30] |
| Downscale 50% | [INSERT: rob_acc_ds50] | [INSERT: rob_f1_ds50] | [INSERT: rob_auc_ds50] |
| Downscale 25% | [INSERT: rob_acc_ds25] | [INSERT: rob_f1_ds25] | [INSERT: rob_auc_ds25] |
| Gaussian blur σ=1 | [INSERT: rob_acc_blur1] | [INSERT: rob_f1_blur1] | [INSERT: rob_auc_blur1] |
| Gaussian blur σ=2 | [INSERT: rob_acc_blur2] | [INSERT: rob_f1_blur2] | [INSERT: rob_auc_blur2] |

> Điền từ `reports/robustness/` khi có kết quả.  
> **Hình 5.4** — Biểu đồ đường robustness theo mức độ nhiễu: `reports/robustness/robustness_chart.png`

---

### 5.4.4 Nhận Xét

#### Đánh Giá Tổng Quan

Mô hình EfficientNet-B0 sau Giai đoạn 3 đạt kết quả xuất sắc trên tập test chuẩn với Accuracy 97,33%, F1-Score 97,45%, và AUC-ROC 99,70%. Chiến lược progressive unfreezing chứng minh hiệu quả rõ rệt: mỗi giai đoạn mở băng thêm đều đóng góp cải thiện đo được, với tổng mức tăng +10,71 điểm phần trăm accuracy từ Giai đoạn 1 đến Giai đoạn 3 trên tập val.

AUC-ROC 0,9970 có ý nghĩa đặc biệt: đây là độ đo độc lập với ngưỡng phân loại, cho thấy mô hình có khả năng phân tách xác suất thật/giả rất tốt ở mọi mức ngưỡng — đảm bảo tính linh hoạt khi triển khai thực tế với các yêu cầu precision/recall khác nhau.

#### Đánh Giá Theo Từng Độ Đo

**Precision (0,9773):** Khi mô hình dự đoán một ảnh là giả, xác suất đúng đạt 97,73%. Precision cao bác bỏ giả thuyết mô hình thiên vị về lớp đa số (53% Fake) — nếu mô hình đoán thiên về Fake, số False Positive tăng và Precision giảm; kết quả thực tế cho thấy điều ngược lại.

**Recall (0,9718):** Mô hình phát hiện được 97,18% tổng số ảnh giả thực tế. Khoảng cách nhỏ giữa Precision (97,73%) và Recall (97,18%) — chênh lệch 0,55 điểm phần trăm — cho thấy mô hình không thiên lệch đáng kể về phía nào.

**F1-Score (0,9745):** Trung bình điều hòa cân bằng giữa Precision và Recall, xác nhận mô hình hoạt động tốt đều trên cả hai phương diện phát hiện.

#### Phân Tích Nguyên Nhân Thành Công

Hiệu suất cao đạt được nhờ sự kết hợp của nhiều yếu tố:

- **Quy mô dữ liệu đa nguồn:** 316.530 ảnh từ 4 phương pháp sinh khác nhau (StyleGAN2, manipulation-based, GAN bậc cao, ciplab) buộc mô hình học đặc trưng phổ quát thay vì phụ thuộc vào artifact của một phương pháp cụ thể.
- **Progressive unfreezing:** Ngăn catastrophic forgetting, cho phép backbone thích nghi dần với domain khuôn mặt mà không phá vỡ đặc trưng ImageNet ban đầu.
- **Mixed precision FP16:** Cho phép batch size 128 trên VRAM 12GB, cung cấp gradient estimate ổn định hơn so với batch nhỏ.
- **EarlyStopping không kích hoạt:** Cho thấy 30 epochs là mức hợp lý — mô hình chưa overfitting khi kết thúc huấn luyện; tăng thêm epochs có thể cải thiện thêm.

#### Giới Hạn Quan Sát Được

- **EarlyStopping không kích hoạt ở cả ba stage:** Là tín hiệu tích cực nhưng cũng gợi ý mô hình có thể đạt kết quả tốt hơn nếu tăng số epoch.
- **Chưa có kết quả cross-generator:** Chưa thể đánh giá mức độ tổng quát hóa sang ảnh giả từ phương pháp chưa thấy trong training (Diffusion models, Stable Diffusion, Midjourney).
- **Chưa có kết quả robustness:** Chưa biết hiệu suất suy giảm như thế nào khi ảnh bị nén JPEG mạnh (q=30) hoặc blur — quan trọng cho triển khai thực tế.

---

## 5.5 Ứng Dụng Thực Nghiệm (Demo)

### 5.5.1 Pipeline Inference

Quá trình phân loại một ảnh mới được thực hiện theo pipeline:

```
Ảnh đầu vào (định dạng bất kỳ)
    → PIL.Image.open() → convert("RGB")
    → albumentations: Resize(224,224) → Normalize(ImageNet mean/std) → ToTensorV2
    → unsqueeze(0)                        # Thêm batch dimension
    → model.eval(); model(tensor)         # Forward pass
    → torch.sigmoid(logit)               # Chuyển logit → xác suất [0,1]
    → "Fake" nếu prob > 0.5, "Real" nếu prob ≤ 0.5
```

Ngưỡng mặc định là 0,5. Trong ứng dụng thực tế với yêu cầu Recall cao hơn (ưu tiên không bỏ sót ảnh giả), ngưỡng có thể được hạ xuống (ví dụ: 0,3–0,4) dựa trên đường ROC.

### 5.5.2 Script Demo

Script `scripts/run_evaluate.py` hỗ trợ đánh giá checkpoint riêng lẻ trên tập test. [INSERT: mô tả script demo.py nếu đã implement — giao diện dòng lệnh, đầu vào/đầu ra, ví dụ chạy]

### 5.5.3 Grad-CAM — Trực Quan Hóa Vùng Quyết Định

[INSERT: kết quả Grad-CAM từ `artifacts/gradcam/` khi có]

Grad-CAM (Gradient-weighted Class Activation Mapping) tạo ra bản đồ nhiệt (heatmap) làm nổi bật các vùng ảnh đóng góp nhiều nhất vào quyết định của mô hình. Dựa trên quan sát EDA (Phần 3.2.3), kỳ vọng mô hình tập trung vào các vùng:
- Vùng tai và hoa tai — nơi StyleGAN2 thường để lại artifact rõ nhất.
- Vùng tiếp giáp tóc–da — ranh giới khó tổng hợp tự nhiên.
- Nền ảnh gần mặt — thường bị biến dạng trong quá trình ghép khuôn mặt (manipulation-based).

> **Hình 5.5** — Ví dụ Grad-CAM overlay: ảnh thật (trái) và ảnh giả (phải)  
> Nguồn: `artifacts/gradcam/` [INSERT khi có]
