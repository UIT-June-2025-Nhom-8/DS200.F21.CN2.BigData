# Chương 7 — Hướng Phát Triển Đề Tài

> **Cập nhật:** 28/05/2026

---

Dựa trên kết quả thực nghiệm và các giới hạn được xác định trong quá trình nghiên cứu, sáu hướng phát triển được đề xuất cho các nghiên cứu tiếp theo:

**1. Mở rộng sang ảnh từ diffusion models thế hệ mới:**  
Tập dữ liệu hiện tại chủ yếu bao gồm ảnh GAN (StyleGAN2, manipulation-based). Các mô hình khuếch tán (Stable Diffusion, Midjourney, DALL-E 3) tạo ra artifact hoàn toàn khác về cấu trúc tần số. Mở rộng tập huấn luyện với ảnh từ các generator này — theo hướng của GenImage [Zhu et al., NeurIPS 2023] và AI-Face [Lin et al., CVPR 2025] — là bước cần thiết để duy trì hiệu suất trước làn sóng công cụ sinh ảnh mới.

**2. Thử nghiệm kiến trúc backbone mạnh hơn:**  
EfficientNet-B0 đạt kết quả tốt nhờ sự cân bằng giữa tốc độ và độ chính xác. Các hướng mở rộng đáng thử bao gồm: EfficientNet-B4/B7 (cùng họ, độ sâu cao hơn), Vision Transformer (ViT-B/16) với cơ chế attention toàn cục có thể bắt được artifact phân tán không gian, và hybrid architectures kết hợp CNN với attention.

**3. Phát hiện deepfake trong video:**  
Pipeline hiện tại xử lý ảnh tĩnh đơn lẻ. Mở rộng sang video đòi hỏi mô hình hóa tính nhất quán thời gian (temporal consistency): khuôn mặt deepfake trong video thường có nhấp nháy (flickering) hoặc không nhất quán giữa các frame liên tiếp. Kiến trúc LSTM hoặc Transformer trên chuỗi frame là hướng tiếp cận tự nhiên.

**4. Tối ưu hóa cho triển khai thực tế:**  
Mô hình Stage 3 hiện có kích thước 52,5 MB — phù hợp cho server nhưng nặng cho thiết bị biên. Các kỹ thuật nén mô hình (quantization INT8, structured pruning, knowledge distillation sang EfficientNet-Lite) có thể giảm kích thước 4–8 lần với mức giảm hiệu suất tối thiểu, mở đường cho triển khai trên mobile/edge.

**5. Đánh giá robustness với adversarial examples:**  
Bài kiểm tra robustness hiện tại giới hạn ở nhiễu tự nhiên (JPEG, blur). Trong kịch bản tấn công có chủ đích, kẻ xấu có thể tạo perturbation tối thiểu (FGSM, PGD, C&W) để đánh lừa mô hình phân loại ảnh giả thành thật. Đánh giá adversarial robustness và huấn luyện với adversarial training là bước quan trọng để hardening hệ thống cho ứng dụng bảo mật thực tế.

**6. Tích hợp cơ chế xác thực nguồn gốc ảnh (C2PA):**  
Hướng tiếp cận phát hiện dựa trên đặc trưng thị giác (như đề tài này) luôn là cuộc chạy đua vũ trang với các generator ngày càng tốt hơn. Kết hợp với chuẩn C2PA (Coalition for Content Provenance and Authenticity) — gắn metadata chứng thực nguồn gốc vào ảnh tại thời điểm chụp — tạo ra cơ chế phòng thủ bổ sung và bền vững hơn về lâu dài.
