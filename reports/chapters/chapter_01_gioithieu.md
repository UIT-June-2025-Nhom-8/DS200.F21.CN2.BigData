# Chương 1 — Giới Thiệu Đề Tài

> **Cập nhật:** 28/05/2026

---

## 1.1 Tính Cấp Thiết Của Đề Tài

Sự phát triển vượt bậc của các mô hình sinh ảnh bằng trí tuệ nhân tạo — đặc biệt là mạng đối nghịch sinh (Generative Adversarial Network — GAN) và các mô hình khuếch tán (Diffusion Models) — đã tạo ra một thách thức nghiêm trọng đối với tính xác thực của hình ảnh số. Từ StyleGAN2 tạo ra khuôn mặt người hoàn toàn không tồn tại ngoài đời thực, đến các công cụ deepfake cho phép ghép khuôn mặt vào video với chất lượng ngày càng cao, ranh giới giữa ảnh thật và ảnh giả ngày càng trở nên mờ nhạt.

Hệ quả xã hội của hiện tượng này là rất đáng lo ngại. Ảnh khuôn mặt giả mạo đã được sử dụng trong các chiến dịch thao túng dư luận chính trị, tạo hình đại diện giả mạo cho các tài khoản lừa đảo trên mạng xã hội, và làm vũ khí trong các vụ xâm phạm danh dự cá nhân. Nghiên cứu của Bird & Lotfi (2024) và Yan và cộng sự (2025) chỉ ra rằng ngay cả những người được đào tạo về kỹ thuật số cũng khó phân biệt khuôn mặt do AI tổng hợp bằng mắt thường, đặc biệt với các mô hình thế hệ mới. Điều này đặt ra nhu cầu cấp bách về các công cụ phát hiện tự động có độ chính xác cao và khả năng tổng quát hóa sang các phương pháp tổng hợp mới.

Thách thức kỹ thuật không kém phần phức tạp. Mỗi phương pháp tổng hợp ảnh để lại các loại artifact khác nhau: StyleGAN2 thường tạo ra artifact tần số cao ở vùng tai và tóc; deepfake dựa trên ghép khuôn mặt thường có vùng biên không tự nhiên; diffusion models tạo ra artifact hoàn toàn khác về cấu trúc tần số. Một mô hình phát hiện huấn luyện chủ yếu trên ảnh StyleGAN có thể đạt accuracy cao trên bộ test cùng phân bố nhưng thất bại khi gặp ảnh từ generator mới — vấn đề domain generalization được ghi nhận bởi Zhu và cộng sự (NeurIPS 2023) và Li và cộng sự (CVPR 2020).

Trong bối cảnh đó, việc xây dựng một hệ thống phân loại khuôn mặt thật/giả dựa trên học sâu với khả năng tổng quát hóa đa nguồn là mục tiêu có giá trị khoa học và ứng dụng thực tiễn rõ ràng.

---

## 1.2 Mô Tả Bài Toán

Đề tài đặt ra bài toán phân loại nhị phân ảnh khuôn mặt:

- **Đầu vào (Input):** Một ảnh khuôn mặt đơn lẻ định dạng RGB, kích thước bất kỳ (sau bước resize về 224×224).
- **Đầu ra (Output):** Nhãn nhị phân — $y \in \{0, 1\}$ với $0$ = thật (Real), $1$ = giả (Fake) — kèm theo xác suất tin cậy $p \in [0, 1]$.

Bài toán được định nghĩa chính thức là: tìm hàm $f: \mathcal{X} \rightarrow [0,1]$ sao cho $f(\mathbf{x}) > 0.5$ khi và chỉ khi ảnh $\mathbf{x}$ là ảnh khuôn mặt do AI tổng hợp, trong đó $\mathcal{X}$ là không gian ảnh RGB 224×224.

Ràng buộc thực tế quan trọng: mô hình phải hoạt động trên ảnh từ nhiều phương pháp tổng hợp khác nhau (StyleGAN2, manipulation-based, GAN tổng quát, ciplab), không chỉ tối ưu trên một phân bố dữ liệu cụ thể. Đây là ràng buộc phân biệt đề tài này với nhiều nghiên cứu chỉ đánh giá trên một bộ dữ liệu đơn nguồn.

Các độ đo đánh giá được sử dụng gồm: Accuracy, Precision, Recall, F1-Score và AUC-ROC, tính trên tập test độc lập (15% toàn bộ dữ liệu, 47.480 ảnh).

---

## 1.3 Đối Tượng và Phạm Vi Nghiên Cứu

**Đối tượng nghiên cứu:**
- Ảnh khuôn mặt người thật và khuôn mặt do AI tổng hợp (GAN, manipulation-based).
- Kiến trúc EfficientNet-B0 trong bài toán phân loại ảnh nhị phân với transfer learning.
- Chiến lược fine-tuning 3 giai đoạn (progressive unfreezing) và ảnh hưởng của từng giai đoạn đến hiệu suất.
- Tăng cường dữ liệu (data augmentation) bằng albumentations và vai trò của nó trong khả năng tổng quát hóa.

**Phạm vi nghiên cứu:**
- *Dữ liệu:* Bốn bộ Kaggle dưới `data/raw/` (140k-StyleGAN, Deepfake-Real, Hard-FakeReal, ciplab). Tổng 316.530 ảnh sau dedup, phân chia stratified 70/15/15.
- *Phương pháp:* Transfer learning từ EfficientNet-B0 pretrained trên ImageNet, huấn luyện 3 giai đoạn với mixed-precision FP16.
- *Đánh giá:* Test set chuẩn (in-distribution) + cross-generator test (out-of-distribution) + robustness test (JPEG compression, downscale, Gaussian blur) + Grad-CAM explainability.
- *Không trong phạm vi:* Video deepfake (chỉ xử lý ảnh tĩnh), audio deepfake, real-time detection, ảnh từ diffusion models thế hệ mới (Stable Diffusion, Midjourney) — đây là hạn chế và hướng phát triển tương lai.

---

## 1.4 Mục Tiêu Đề Tài

Đề tài đặt ra tám mục tiêu cụ thể và đo lường được:

1. **Xây dựng pipeline dữ liệu:** Thu thập và hợp nhất bốn bộ Kaggle, loại bỏ trùng lặp MD5, tạo file phân chia `data/splits/train.csv`, `val.csv`, `test.csv` theo tỷ lệ 70/15/15 stratified.

2. **Thực hiện EDA toàn diện:** Phân tích phân bố lớp theo nguồn, phân bố độ phân giải, thống kê pixel R/G/B, phân tích phổ FFT để xác định đặc trưng phân biệt thật/giả.

3. **Xây dựng pipeline tiền xử lý:** Chuẩn hóa resize 224×224, normalize ImageNet, augmentation tách biệt train/val với albumentations; đảm bảo tái lập đầy đủ qua seed 42.

4. **Huấn luyện EfficientNet-B0 theo 3 giai đoạn:** Head-only (LR=1e-4) → partial unfreeze blocks 5–6 (LR=1e-5) → full fine-tune (LR=1e-6); lưu checkpoint tốt nhất theo val\_loss mỗi giai đoạn.

5. **Đánh giá trên test set chuẩn:** Tính bảng Accuracy/Precision/Recall/F1/AUC-ROC; vẽ confusion matrix và ROC curve; mục tiêu AUC-ROC ≥ 0,95.

6. **Cross-generator evaluation:** Huấn luyện trên 3 bộ, đánh giá trên bộ còn lại (hold-out), lặp lại cho cả 4 bộ; ghi nhận mức độ suy giảm hiệu suất.

7. **Robustness evaluation:** Đánh giá với JPEG compression (q=70/50/30), downscale (50%/25%), Gaussian blur (σ=1/2); vẽ biểu đồ đường accuracy vs mức nhiễu.

8. **Grad-CAM explainability:** Trực quan hóa vùng ảnh mô hình tập trung khi phân loại; phân tích tương quan với artifact quan sát được trong EDA.