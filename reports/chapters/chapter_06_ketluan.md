# Chương 6 — Kết Luận

> **Cập nhật:** 28/05/2026

---

## 6.1 Tóm Tắt Đề Tài

Đề tài nghiên cứu bài toán phân loại nhị phân ảnh khuôn mặt thật và ảnh khuôn mặt do AI tổng hợp — một vấn đề kỹ thuật có ý nghĩa xã hội cấp thiết trong bối cảnh các công cụ sinh ảnh bằng GAN và diffusion models ngày càng tạo ra nội dung photorealistic khó phân biệt bằng mắt thường. Đề tài xây dựng pipeline hoàn chỉnh từ thu thập và hợp nhất bốn bộ dữ liệu Kaggle đa dạng (316.530 ảnh sau dedup) đến huấn luyện mô hình EfficientNet-B0 theo chiến lược fine-tuning 3 giai đoạn (progressive unfreezing), đạt Accuracy 97,33%, F1-Score 97,45% và AUC-ROC 99,70% trên tập test độc lập. Ngoài đánh giá chuẩn, đề tài thiết kế thêm bài kiểm tra cross-generator để đo khả năng tổng quát hóa sang nguồn dữ liệu chưa thấy trong huấn luyện, bài kiểm tra robustness để đánh giá độ bền dưới nhiễu JPEG/blur/downscale, và trực quan hóa Grad-CAM để giải thích cơ chế quyết định của mô hình.

---

## 6.2 Đóng Góp Của Đề Tài

Đề tài thực hiện năm đóng góp cụ thể:

1. **Pipeline dữ liệu đa nguồn quy mô lớn:** Xây dựng quy trình hợp nhất bốn bộ dữ liệu Kaggle (StyleGAN2, manipulation-based, GAN bậc cao, ciplab) với dedup MD5 và stratified split 70/15/15, tạo ra tập huấn luyện 221.571 ảnh đa dạng hơn so với các nghiên cứu thường chỉ sử dụng một nguồn đơn lẻ.

2. **Chiến lược huấn luyện 3 giai đoạn có kiểm soát:** Áp dụng progressive unfreezing (Head-only → Partial unfreeze blocks 5–6 → Full fine-tune) kết hợp EarlyStopping và ReduceLROnPlateau, với toàn bộ siêu tham số được quản lý tập trung qua `configs/train_config.yaml` đảm bảo tái lập hoàn toàn. Val accuracy cải thiện rõ rệt qua từng giai đoạn: 86,63% → 94,54% → 97,34%.

3. **Đánh giá cross-generator có hệ thống:** Thực hiện hold-out evaluation theo từng bộ dữ liệu nguồn để định lượng khả năng tổng quát hóa thực tế — quan trọng hơn benchmark in-distribution thông thường.

4. **Đánh giá robustness với nhiễu thực tế:** Kiểm tra độ bền với JPEG compression (q=70/50/30), downscale (50%/25%) và Gaussian blur (σ=1/2), phản ánh điều kiện ảnh phổ biến khi chia sẻ qua mạng xã hội.

5. **Explainability qua Grad-CAM:** Cung cấp bằng chứng trực quan về vùng ảnh mô hình tập trung khi phân loại, tăng tính minh bạch và hỗ trợ tin cậy hóa hệ thống AI trong bối cảnh ứng dụng thực tế.
