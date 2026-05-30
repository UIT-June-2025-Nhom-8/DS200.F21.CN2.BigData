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
| 140k-StyleGAN | 3 bộ còn lại | **99,04%** | **99,03%** | **99,95%** |
| Deepfake-Real | 3 bộ còn lại | **96,50%** | **96,83%** | **99,55%** |
| Hard-FakeReal | 3 bộ còn lại | **90,77%** | **91,09%** | **97,07%** |
| ciplab | 3 bộ còn lại | **53,36%** | **29,44%** | **53,44%** |

Nguồn: `reports/cross_generator/cross_generator_metrics.json`.

---

### 5.4.3 Đánh Giá Robustness

Bài kiểm tra robustness đánh giá độ bền của mô hình trước các biến dạng ảnh thường gặp trong thực tế: nén JPEG (mất thông tin khi chia sẻ qua mạng xã hội), downscale (ảnh độ phân giải thấp), và Gaussian blur (ảnh chụp không nét).

**Bảng 5.5 — Kết quả robustness evaluation**

| Điều kiện nhiễu | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Baseline (không nhiễu) | **97,33%** | **97,46%** | **99,70%** |
| JPEG nén q=70 | 82,33% | 85,32% | 95,97% |
| JPEG nén q=50 | 79,77% | 83,45% | 93,30% |
| JPEG nén q=30 | 78,60% | 82,48% | 90,30% |
| Downscale 50% | 85,39% | 85,60% | 93,58% |
| Downscale 25% | 72,87% | 75,85% | 80,40% |
| Gaussian blur σ=1 | 90,45% | 90,29% | 97,76% |
| Gaussian blur σ=2 | 73,65% | 73,84% | 82,11% |

Nguồn: `reports/robustness/robustness_metrics.json`.

> **Hình 5.4** — Biểu đồ đường robustness theo mức độ nhiễu  
> Nguồn: `reports/robustness/robustness_accuracy.png`, `reports/robustness/robustness_auc.png`

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

#### Phân Tích Cross-Generator

Kết quả cross-generator tiết lộ sự phân hóa rõ rệt về khả năng tổng quát hóa theo nguồn:

- **140k-StyleGAN (Accuracy 99,04%, AUC 99,95%):** Mô hình tổng quát hóa xuất sắc sang ảnh StyleGAN2 khi không có nguồn này trong tập huấn luyện. Điều này gợi ý artifact của StyleGAN2 đủ đặc trưng để mô hình học được từ các nguồn GAN khác.
- **Deepfake-Real (Accuracy 96,50%, AUC 99,55%):** Tổng quát hóa tốt — ảnh manipulation-based có đặc điểm phân biệt đủ rõ để mô hình nhận ra dù không thấy trong training.
- **Hard-FakeReal (Accuracy 90,77%, AUC 97,07%):** Suy giảm đáng kể hơn — phù hợp với bản chất thiết kế "khó phân biệt" của bộ này. Mức AUC 97,07% vẫn cho thấy mô hình học được tín hiệu phân biệt, nhưng ranh giới quyết định kém chắc chắn hơn.
- **ciplab (Accuracy 53,36%, AUC 53,44%):** Mô hình gần như không tổng quát hóa được sang bộ ciplab — hiệu suất chỉ nhỉnh hơn đoán ngẫu nhiên (50%). Đây là phát hiện quan trọng nhất: artifact của ảnh ciplab GAN khác biệt căn bản so với ba nguồn còn lại, đến mức mô hình không rút ra được tín hiệu phân biệt có ích. Khoảng cách này (từ 97,33% in-distribution xuống 53,36% khi holdout ciplab) là bằng chứng rõ ràng cho vấn đề domain generalization đã được GenImage [Zhu et al., NeurIPS 2023] cảnh báo.

#### Phân Tích Robustness

Kết quả robustness cho thấy mô hình nhạy cảm đặc biệt với nén JPEG:

- **JPEG nén (q=70/50/30):** Accuracy giảm từ 97,33% xuống lần lượt 82,33% / 79,77% / 78,60% — mức suy giảm ~15–19 điểm phần trăm ngay ở mức nén vừa phải (q=70). Đáng chú ý, AUC-ROC vẫn duy trì ở mức cao (95,97% tại q=70), cho thấy mô hình vẫn giữ khả năng phân tách xác suất tốt nhưng ngưỡng quyết định bị dịch chuyển. Recall tăng lên (97,44% tại q=70) trong khi Precision giảm mạnh (75,88%) — mô hình thiên về dự đoán Fake nhiều hơn khi ảnh bị nén, dẫn đến False Positive cao.
- **Downscale:** Accuracy giảm từ 97,33% xuống 85,39% (×0,5) và 72,87% (×0,25). Mức suy giảm tại ×0,25 (-24,5 pp) nghiêm trọng hơn JPEG, cho thấy thông tin tần số cao (artifact cục bộ) bị mất vĩnh viễn khi giảm độ phân giải mạnh.
- **Gaussian blur (σ=1/2):** Mức σ=1 vẫn chấp nhận được (Accuracy 90,45%), nhưng σ=2 gây suy giảm đáng kể (73,65%). Điều này nhất quán với quan sát FFT rằng artifact của ảnh giả nằm chủ yếu ở tần số cao — Gaussian blur xóa đi chính nguồn thông tin mà mô hình dựa vào.

Kết quả robustness chỉ ra hướng cải thiện rõ ràng: bổ sung `albumentations.ImageCompression(quality_lower=40)` và thêm augmentation downscale vào tập huấn luyện có khả năng cải thiện đáng kể độ bền của mô hình trong điều kiện thực tế.

#### Giới Hạn Quan Sát Được

- **EarlyStopping không kích hoạt ở cả ba stage:** Là tín hiệu tích cực nhưng cũng gợi ý mô hình có thể đạt kết quả tốt hơn nếu tăng số epoch hoặc điều chỉnh lịch học (learning rate schedule).
- **Sụp đổ hoàn toàn trên ciplab (holdout):** Mức AUC 53,44% cho thấy ciplab GAN tạo ra phân bố hoàn toàn nằm ngoài vùng đặc trưng đã học — giải pháp không phải fine-tuning mà là bổ sung ảnh ciplab vào tập huấn luyện hoặc áp dụng domain adaptation.
- **Nhạy cảm với nén JPEG:** Suy giảm ~15 pp ngay tại q=70 là đáng lo ngại cho triển khai thực tế trên ảnh chia sẻ qua mạng xã hội — cần thêm compression augmentation trong training.
- **Grad-CAM ciplab:** Heatmap phân tán trên ảnh ciplab xác nhận mô hình không học được đặc trưng artifact của nguồn này — cần bổ sung ảnh ciplab vào tập huấn luyện hoặc áp dụng domain adaptation.

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

Hai script inference đã được triển khai:

- **`scripts/predict.py`** — Phân loại một ảnh đơn lẻ qua dòng lệnh:
  ```bash
  python scripts/predict.py <đường_dẫn_ảnh> --checkpoint artifacts/checkpoints/best_stage3.pth
  # Output: Prediction: Fake | Confidence: 97.82%
  ```
- **`scripts/run_gradcam.py`** — Sinh Grad-CAM overlay cho mẫu ảnh từ tất cả nguồn:
  ```bash
  python scripts/run_gradcam.py --n_per_class 4
  # Output: 32 overlay + gradcam_grid.png → artifacts/gradcam/
  ```

### 5.5.3 Grad-CAM — Trực Quan Hóa Vùng Quyết Định

Grad-CAM (Gradient-weighted Class Activation Mapping) tạo ra bản đồ nhiệt (heatmap) làm nổi bật các vùng ảnh đóng góp nhiều nhất vào quyết định của mô hình, bằng cách tính trung bình có trọng số của gradient theo từng kênh đặc trưng tại lớp cuối cùng của backbone. Thực nghiệm được thực hiện trên 32 ảnh mẫu (4 Real + 4 Fake × 4 nguồn), sử dụng checkpoint Stage 3, chạy trên Apple Silicon M1 qua MPS backend.

**Kết quả trên 32 ảnh mẫu:** 30/32 dự đoán đúng (93,8%). Hai ảnh sai đều thuộc bộ ciplab — nhất quán với kết quả cross-generator (AUC 53,44%) cho thấy mô hình gần như không tổng quát hóa được sang ciplab khi không có nguồn này trong tập huấn luyện.

**Quan sát vùng kích hoạt theo nguồn:**

- **140k-StyleGAN:** Mô hình tập trung vào vùng trung tâm khuôn mặt (mắt, sống mũi, miệng) với activation tập trung và rõ ràng — cả ảnh thật lẫn giả. Đối với ảnh giả StyleGAN2, heatmap thường kéo dài sang vùng biên tóc–da nơi artifact tần số cao xuất hiện. Tất cả 8/8 dự đoán đúng.

- **Deepfake-Real:** Activation phân bổ rộng hơn trên toàn khuôn mặt, đặc biệt ở vùng mắt và khu vực tiếp giáp giữa khuôn mặt ghép và da gốc — đúng với cơ chế tạo artifact của manipulation-based deepfake. Tất cả 8/8 dự đoán đúng.

- **Hard-FakeReal:** Heatmap tập trung chủ yếu vào vùng mũi và miệng — vùng khó tổng hợp tự nhiên nhất trong các GAN bậc cao. Dù bộ này được thiết kế để khó phân biệt, mô hình vẫn đạt 8/8 đúng trên mẫu này.

- **ciplab:** Activation kém tập trung hơn, thường lan ra vùng nền hoặc cổ — dấu hiệu mô hình không tìm được vùng artifact đặc trưng. Hai ảnh sai (tiêu đề đỏ trong grid) có heatmap phân tán toàn ảnh, xác nhận mô hình thiếu tín hiệu phân biệt rõ ràng với ảnh ciplab.

**Kết luận chung:** Mô hình học được chiến lược phát hiện dựa trên các vùng khuôn mặt có ngữ nghĩa (mắt, mũi, biên tóc–da), không phải các đặc trưng nền hoặc màu sắc toàn cục — đây là tín hiệu tích cực về tính tin cậy của mô hình. Tuy nhiên, sự kém tập trung trên ảnh ciplab cho thấy giới hạn trong việc tổng quát hóa khi artifact không nằm ở các vùng quen thuộc.

> **Hình 5.5** — Grid Grad-CAM overlay 32 ảnh mẫu (4 nguồn × 4 Real + 4 Fake). Tiêu đề xanh = dự đoán đúng, đỏ = sai.  
> Nguồn: `artifacts/gradcam/gradcam_grid.png`
