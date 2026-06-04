# Chương 2 — Nghiên Cứu Liên Quan

> **Nguồn tham khảo chính:** `task_plan_ai_face_detection.md`  
> **Cập nhật:** 28/05/2026

---

## 2.1 Các Nghiên Cứu Về Phát Hiện Khuôn Mặt Giả Mạo

Phát hiện khuôn mặt giả mạo (face forgery detection) đã trở thành một nhánh nghiên cứu tích cực trong cộng đồng thị giác máy tính kể từ khi các phương pháp tổng hợp ảnh bằng mạng đối nghịch sinh (GAN) trở nên phổ biến. Các hướng tiếp cận chính có thể phân loại thành ba nhóm: phân tích miền tần số, phân tích texture cục bộ, và học đặc trưng bằng CNN sâu.

**FaceForensics++** [1] là một trong những bộ dữ liệu và phương pháp chuẩn quan trọng nhất trong lĩnh vực này. Rössler và cộng sự (ICCV 2019) xây dựng bộ dữ liệu gồm hơn 1,8 triệu frame từ 1.000 video gốc, áp dụng bốn phương pháp thao túng khuôn mặt (DeepFakes, Face2Face, FaceSwap, NeuralTextures) và huấn luyện nhiều kiến trúc CNN (XceptionNet, MesoNet) để phát hiện. Kết quả cho thấy các mô hình CNN sâu vượt trội so với phương pháp trích xuất đặc trưng thủ công trong điều kiện ảnh chất lượng cao, nhưng hiệu suất suy giảm đáng kể khi ảnh bị nén JPEG — gợi ý về vấn đề robustness mà nghiên cứu này cũng cần giải quyết.

**Celeb-DF** [2] (Li và cộng sự, CVPR 2020) đặt ra thách thức cao hơn với bộ dữ liệu deepfake thế hệ mới: 5.639 video deepfake chất lượng cao từ 59 người nổi tiếng, với chất lượng tốt hơn nhiều so với FaceForensics++. Nghiên cứu này chứng minh rằng nhiều mô hình đạt AUC cao trên FaceForensics++ nhưng lại suy giảm mạnh trên Celeb-DF, làm nổi bật vấn đề generalization cross-dataset — tương tự thử thách cross-generator mà đề tài này đặt ra.

**CIFAKE** [3] (Bird & Lotfi, IEEE Access 2024) nghiên cứu phân loại ảnh do AI tổng hợp bằng diffusion models so với ảnh thật, sử dụng CNN và phân tích explainability (LIME, SHAP) để giải thích quyết định của mô hình. Kết quả cho thấy CNN đơn giản có thể đạt accuracy > 90% nhưng đặc trưng quyết định thường là các artifact texture tinh vi, không phải nội dung ngữ nghĩa — phù hợp với quan sát FFT trong EDA của đề tài này.

**GenImage** [4] (Zhu và cộng sự, NeurIPS 2023) xây dựng benchmark triệu ảnh từ 8 mô hình sinh ảnh khác nhau (Stable Diffusion, Midjourney, DALL-E, v.v.) để đánh giá khả năng tổng quát hóa cross-generator. Nghiên cứu chỉ ra rằng mô hình huấn luyện trên ảnh từ một hoặc vài generator thường suy giảm đáng kể khi gặp generator mới — xác nhận tầm quan trọng của bộ dữ liệu đa nguồn như cách tiếp cận của đề tài này.

---

## 2.2 Các Nghiên Cứu Về Tổng Hợp Khuôn Mặt Bằng AI

**StyleGAN2** (Karras và cộng sự, 2020) là kiến trúc GAN tạo khuôn mặt photorealistic chất lượng cao nhất tính đến thời điểm xuất bản, dựa trên cơ chế style-based generator: phân tách thông tin content và style ở từng mức độ phân giải độc lập. Khuôn mặt được tổng hợp bởi StyleGAN2 (chiếm 70.000/316.530 ảnh trong tập dữ liệu của đề tài) rất khó phân biệt bằng mắt thường, đặt ra thách thức lớn cho phương pháp phân loại truyền thống. Các artifact đặc trưng của StyleGAN2 (vùng tai, tóc, hoa tai) được phát hiện qua EDA và là mục tiêu học tập quan trọng của mô hình.

**AI-Face** [5] (Lin và cộng sự, CVPR 2025) cung cấp bộ dữ liệu và benchmark đánh giá sự công bằng (fairness) trong phát hiện khuôn mặt AI-generated, bao gồm ảnh từ nhiều phương pháp sinh thế hệ mới (diffusion models). Nghiên cứu chỉ ra rằng mô hình phát hiện thường có thiên vị về sắc tộc và giới tính — một hạn chế cần lưu ý khi đánh giá mô hình trên các bộ dữ liệu đa dạng như Hard-FakeReal.

**AIDE** [6] (Yan và cộng sự, ICLR 2025) đề xuất framework kiểm tra độ tin cậy (sanity check) cho các phương pháp phát hiện ảnh AI-generated, chỉ ra rằng nhiều mô hình đạt kết quả cao nhờ học các đặc trưng không bền vững (non-robust features) dễ bị đánh lừa. Kết quả này nhấn mạnh tầm quan trọng của bài kiểm tra robustness (nén JPEG, blur, downscale) mà đề tài này triển khai để đánh giá tính ổn định thực tế của mô hình.

---

## 2.3 Các Nghiên Cứu Về EfficientNet và Transfer Learning

**EfficientNet** (Tan & Le, ICML 2019) giới thiệu phương pháp *compound scaling* để mở rộng mạng CNN: thay vì chỉ tăng độ sâu (depth), độ rộng (width) hoặc độ phân giải (resolution) một chiều như các phương pháp trước, EfficientNet scale đồng thời cả ba chiều theo một hệ số tỷ lệ thống nhất được tìm bằng Neural Architecture Search. EfficientNet-B0 — phiên bản nhỏ nhất — đạt accuracy tương đương ResNet-50 với ~5,3M tham số (so với ~25,6M), làm nó trở thành backbone lý tưởng cho fine-tuning trên tập dữ liệu có domain hỗn hợp.

Transfer learning từ mô hình tiền huấn luyện trên ImageNet đã được chứng minh hiệu quả trong nhiều bài toán phân tích ảnh khuôn mặt. Đặc trưng cấp thấp (cạnh, góc, texture) học được từ ImageNet có giá trị tái sử dụng cao trong phân tích artifact của ảnh khuôn mặt giả mạo. Chiến lược progressive unfreezing — đóng băng backbone hoàn toàn trong giai đoạn đầu rồi mở băng dần từ các lớp cao nhất — được chứng minh giảm nguy cơ *catastrophic forgetting* và cải thiện kết quả cuối so với fine-tuning toàn bộ ngay từ đầu [xem CLAUDE.md, Training Stages].

**Roy và cộng sự** [7] (2026) xây dựng bộ dữ liệu toàn diện kết hợp ảnh người thật và ảnh AI-generated từ nhiều nguồn, phân tích đặc điểm phân biệt và so sánh nhiều kiến trúc CNN. Nghiên cứu này xác nhận rằng các kiến trúc được pretrained trên ImageNet (bao gồm EfficientNet) là điểm khởi đầu tốt hơn so với huấn luyện từ đầu (from scratch) cho bài toán phát hiện ảnh AI-generated.

---

## 2.4 Điểm Mới Của Đề Tài

So sánh với các nghiên cứu trên, đề tài này có một số điểm khác biệt và đóng góp cụ thể:

**Bộ dữ liệu đa nguồn quy mô lớn:** Phần lớn nghiên cứu phát hiện deepfake sử dụng một hoặc hai bộ dữ liệu đơn nguồn (FaceForensics++ chỉ dùng video manipulation; CIFAKE chỉ dùng Stable Diffusion). Đề tài này tổng hợp bốn bộ Kaggle đại diện cho bốn phương pháp sinh khác nhau (StyleGAN2, manipulation-based, GAN bậc cao, ciplab), tạo ra tập huấn luyện 316.530 ảnh đa dạng hơn.

**Chiến lược huấn luyện 3 giai đoạn có kiểm soát:** Áp dụng progressive unfreezing với EarlyStopping và ReduceLROnPlateau ở mỗi giai đoạn, đảm bảo fine-tuning tối ưu mà không phá vỡ đặc trưng pretrained. Cấu hình chi tiết được quản lý tập trung qua YAML, đảm bảo tái lập đầy đủ.

**Đánh giá cross-generator có hệ thống:** Thực hiện hold-out evaluation theo từng nguồn dữ liệu để đo khả năng tổng quát hóa thực tế — quan trọng hơn benchmark trên test set cùng phân bố.

**Đánh giá robustness với nhiễu thực tế:** Kiểm tra với JPEG compression (q=70/50/30), downscale (50%/25%) và Gaussian blur (σ=1/2) — phản ánh điều kiện ảnh phổ biến khi chia sẻ qua mạng xã hội hay chụp màn hình.

**Trực quan hóa Grad-CAM:** Cung cấp bằng chứng trực quan về cơ chế quyết định của mô hình, tăng tính minh bạch và tin cậy cho kết quả phân loại.

---

## Tài Liệu Tham Khảo Chương 2

[1] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, và M. Nießner, "FaceForensics++: Learning to Detect Manipulated Facial Images," in *Proc. IEEE/CVF ICCV*, 2019, pp. 1–11.

[2] Y. Li, X. Yang, P. Sun, H. Qi, và S. Lyu, "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics," in *Proc. IEEE/CVF CVPR*, 2020, pp. 3207–3216.

[3] J. J. Bird và A. Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images," *IEEE Access*, 2024.

[4] M. Zhu et al., "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image," in *Proc. NeurIPS*, 2023.

[5] Y. Lin et al., "AI-Face: A Million-Scale Demographically Annotated AI-Generated Face Dataset and Fairness Benchmark," in *Proc. IEEE/CVF CVPR*, 2025.

[6] S. Yan et al., "AIDE: A Sanity Check for AI-generated Image Detection," in *Proc. ICLR*, 2025.

[7] S. Roy et al., "A Comprehensive Dataset for Human vs. AI Generated Image Detection," arXiv:2601.00553, 2026.

[8] M. Tan và Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019, pp. 6105–6114.

[9] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, và T. Aila, "Analyzing and Improving the Image Quality of StyleGAN," in *Proc. IEEE/CVF CVPR*, 2020, pp. 8110–8119.

[10] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in *Proc. IEEE/CVF ICCV*, 2017, pp. 618–626.
