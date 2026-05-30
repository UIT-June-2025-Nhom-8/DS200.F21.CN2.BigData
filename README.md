# Detect AI-Generated Faces

> **Môn học:** DS200.F21.CN2 — Trường ĐH Công nghệ Thông tin, VNU-HCM  
> **Đề tài:** Phát hiện ảnh khuôn mặt tạo bởi AI

---

## Mục Đích

Xây dựng hệ thống phân loại ảnh khuôn mặt **Real/Fake** từ nhiều nguồn dữ liệu, tập trung vào:

- Huấn luyện mô hình học sâu với dữ liệu đa dạng (4 datasets)
- Đánh giá đầy đủ bằng test set riêng
- Kiểm thử khả năng tổng quát hoá và độ bền vững trước nhiễu ảnh
- Trực quan hoá vùng chú ý của mô hình bằng Grad-CAM
- Demo dự đoán qua webapp

---

## Thành Viên

| MSSV | Họ tên | Vai trò |
|---|---|---|
| 25410056 | Lã Xuân Hồng | Dataset & Tiền xử lý |
| 25410150 | Nguyễn Minh Trọng | Dataset & Tiền xử lý |
| 25410034 | Lê Quang Hoài Đức | Triển khai pipeline |
| 25410104 | Nguyễn Minh Nhật | Môi trường giả lập & Triển khai pipeline | 
| 25410088 | Trần Thanh Long | Kiểm thử chất lượng |

---

## Dataset

| # | Tên | Mô tả |
|---|-----|-------|
| 1 | 140k Real and Fake Faces | 70k real (Flickr) + 70k fake (StyleGAN) |
| 2 | Deepfake and Real Images | Ảnh 256×256, manipulation-based |
| 3 | Fake-Vs-Real-Faces (Hard) | Bộ nhỏ, ảnh khó phân biệt |
| 4 | Real and Fake Face Detection | Dataset từ ciplab |

Label được chuẩn hoá về `Real/Fake`, tách train/val/test theo tỉ lệ **70/15/15** (stratified).

---

## Cấu Trúc Dự Án

```text
.
├── artifacts/          # Kết quả: model checkpoint, metrics, Grad-CAM
├── configs/            # Cấu hình train/eval/inference
├── data/
│   ├── raw/            # Dữ liệu gốc (không chỉnh sửa)
│   └── splits/         # CSV train/val/test (image_path, label)
├── notebooks/          # Dataset EDA, training, phân tích kết quả
├── reports/            # Hình ảnh EDA, bảng kết quả, báo cáo
├── scripts/            # Script tiện ích
├── src/                # Mã nguồn chính (dataset, model, train, eval)
└── tests/              # Unit/integration tests
```

> Resize 224×224, normalize ImageNet và augmentation được thực hiện on-the-fly trong PyTorch `Dataset`, không lưu ảnh đã xử lý ra đĩa.

---

## Kế Hoạch Thực Hiện

1. Gộp 4 bộ dữ liệu, chuẩn hoá nhãn, kiểm tra corrupt/duplicate  
2. Tiền xử lý: resize 224x224, normalize theo ImageNet, augment cho train  
3. Huấn luyện EfficientNet-B0 (PyTorch + timm) theo 3 giai đoạn fine-tuning  
4. Đánh giá: Accuracy, Precision, Recall, F1, AUC + Confusion Matrix, ROC  
5. Kiểm thử mở rộng: cross-generator và robustness (JPEG/resize/blur)  
6. Explainability với Grad-CAM và demo webapp bằng Gradio

---

## Tech Stack

| Thành phần | Lựa chọn |
|-----------|---------|
| Framework | PyTorch + timm |
| Backbone | EfficientNet-B0 (pretrained ImageNet) |
| Explainability | pytorch-grad-cam |
| Demo | Gradio |
| GPU | RTX 4070/5070 12GB (local) |

---

## Notebooks

| File | Mô tả |
|------|-------|
| `notebooks/00_00_download_datasets.ipynb` | Tải 4 dataset từ Kaggle (idempotent, tự skip nếu đã có) |
| `notebooks/01_dataset_preparation.ipynb` | Dataset Preparation + basic EDA + split 70/15/15 → lưu CSV |

### Tải Dataset

Mở `00_download_datasets.ipynb`, điền credentials trực tiếp vào cell đầu:

```python
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY      = "your_key"
```

Lấy key tại [kaggle.com/settings](https://www.kaggle.com/settings) → Account → API → Create New Token.  
Chạy cell cleanup cuối notebook để xoá `~/.kaggle/kaggle.json` sau khi tải xong.

---

## Tài Liệu Tham Khảo

- CIFAKE (IEEE Access, 2024): https://arxiv.org/abs/2303.14126  
- GenImage (NeurIPS, 2023): https://arxiv.org/abs/2306.08571  
- AIDE (ICLR, 2025): https://github.com/shilinyan99/AIDE  
- AI-Face (CVPR, 2025): https://arxiv.org/abs/2406.00783  
- FaceForensics++ (ICCV, 2019): https://arxiv.org/abs/1901.08971  
- Celeb-DF (CVPR, 2020): https://arxiv.org/abs/1909.12962
