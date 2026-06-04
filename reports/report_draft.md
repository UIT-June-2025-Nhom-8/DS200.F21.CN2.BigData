# DS200.F21.CN2 — Bản Thảo Báo Cáo Chính Thức

> Tài liệu này là bản thảo cho bài báo khoa học cuối kỳ, tuân theo cấu trúc template BaoCao.docx.  
> Cập nhật lần cuối: 28/05/2026

---

## Chương 1 — Giới Thiệu Đề Tài

### 1.1 Tính Cấp Thiết Của Đề Tài

[INSERT]

### 1.2 Mô Tả Bài Toán

[INSERT]

### 1.3 Đối Tượng và Phạm Vi Nghiên Cứu

[INSERT]

### 1.4 Mục Tiêu Đề Tài

[INSERT]

---

## Chương 2 — Nghiên Cứu Liên Quan

### 2.1 Các Nghiên Cứu Về Phát Hiện Khuôn Mặt Giả Mạo

[INSERT: trích dẫn ≥ 2–3 bài từ task_plan_ai_face_detection.md]

### 2.2 Các Nghiên Cứu Về GAN và Diffusion-Based Image Synthesis

[INSERT: trích dẫn ≥ 2 bài về GAN/diffusion]

### 2.3 Các Nghiên Cứu Về EfficientNet và Transfer Learning

[INSERT: bài gốc EfficientNet + ≥ 1 bài ứng dụng]

### 2.4 Điểm Mới Của Đề Tài

[INSERT]

---

## Chương 3 — Bộ Dữ Liệu

### 3.1 Giới Thiệu và Thu Thập

Nghiên cứu này sử dụng tập dữ liệu tổng hợp từ bốn bộ dữ liệu công khai trên nền tảng Kaggle, mỗi bộ đại diện cho một phương pháp tổng hợp khuôn mặt bằng AI khác nhau. Sự đa dạng về nguồn gốc và phương pháp sinh ảnh là yếu tố then chốt để đảm bảo khả năng tổng quát hóa của mô hình đối với nhiều loại ảnh giả mạo ngoài thực tế.

#### 3.1.1 Mô Tả Các Nguồn Dữ Liệu

Bảng 3.1 trình bày thông tin chi tiết của bốn bộ dữ liệu được sử dụng.

**Bảng 3.1 — Các bộ dữ liệu sử dụng trong nghiên cứu**

| Bộ dữ liệu | Đường dẫn (`data/raw/`) | Quy mô | Phương pháp sinh |
|---|---|---|---|
| 140k-StyleGAN | `140k-real-and-fake-faces/real_vs_fake/real-vs-fake` | 70.000 thật + 70.000 giả | StyleGAN2 |
| Deepfake-Real | `deepfake-and-real-images/Dataset` | [INSERT: số ảnh] | Manipulation-based (ghép khuôn mặt) |
| Hard-FakeReal | `hardfakevsrealfaces` | [INSERT: số ảnh] | GAN tổng hợp (khó phân biệt) |
| ciplab | `real-and-fake-face-detection/real_and_fake_face` | [INSERT: số ảnh] | Nhiều phương pháp GAN |

**Mô tả từng bộ dữ liệu:**

- **140k-StyleGAN:** Bộ dữ liệu lớn nhất trong tập hợp, bao gồm 70.000 ảnh khuôn mặt người thật lấy từ kho ảnh Flickr và 70.000 ảnh giả được tạo bởi StyleGAN2 — một trong những mô hình sinh ảnh GAN tiên tiến nhất tính đến thời điểm xây dựng bộ dữ liệu. Ảnh đã được chuẩn hóa về kích thước 224×224 pixel.

- **Deepfake-Real:** Bộ dữ liệu tập trung vào ảnh được tạo bằng kỹ thuật ghép và biến đổi khuôn mặt (face manipulation), khác biệt về đặc điểm so với ảnh StyleGAN. Ảnh gốc ở kích thước 256×256 pixel.

- **Hard-FakeReal:** Bộ dữ liệu quy mô nhỏ nhưng có mức độ khó cao — ảnh giả trong bộ này được thiết kế để tối đa hóa tính khó phân biệt với mắt thường, bao gồm đa dạng sắc tộc và điều kiện ánh sáng.

- **ciplab:** Bộ dữ liệu từ phòng nghiên cứu ciplab, tổng hợp ảnh giả từ nhiều phương pháp GAN khác nhau, cung cấp sự đa dạng về phong cách và chất lượng ảnh giả.

#### 3.1.2 Quy Trình Hợp Nhất và Tiền Xử Lý

Quá trình xây dựng tập dữ liệu thống nhất được thực hiện qua notebook `notebooks/01_dataset_preparation.ipynb`, bao gồm các bước sau:

**Bước 1 — Suy ra nhãn từ cấu trúc thư mục:**  
Nhãn phân loại được suy ra tự động từ tên thư mục chứa ảnh theo quy tắc:
- Nhãn `'Real'` (thật): thư mục có tên `real`, `Real`, hoặc `training_real`
- Nhãn `'Fake'` (giả): thư mục có tên `fake`, `Fake`, hoặc `training_fake`

**Bước 2 — Loại bỏ ảnh trùng lặp bằng MD5:**  
Mỗi file ảnh được tính giá trị băm MD5 — một "dấu vân tay kỹ thuật số" đặc trưng cho nội dung byte của file. Hai ảnh có cùng giá trị MD5 được xem là hoàn toàn giống nhau và chỉ giữ lại một bản. Bước này ngăn chặn nguy cơ rò rỉ dữ liệu (data leakage) khi cùng một ảnh xuất hiện trong cả tập huấn luyện lẫn tập kiểm tra.

**Bước 3 — Phân chia dữ liệu có tầng (stratified split):**  
Tập dữ liệu sau dedup được phân chia theo tỷ lệ **70% huấn luyện / 15% kiểm định / 15% kiểm tra**, sử dụng phân tầng theo nhãn (`stratified`) với `random_state=42` để đảm bảo tính tái lập. Kết quả được lưu vào ba file CSV:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

**Schema CSV:** Mỗi hàng gồm hai trường — `image_path` (đường dẫn tuyệt đối tới file ảnh, kiểu chuỗi ký tự) và `label` (nhãn chuỗi: `'Real'` = thật, `'Fake'` = giả; được ánh xạ sang `Real=0`, `Fake=1` khi đưa vào mô hình).

#### 3.1.3 Thống Kê Tập Dữ Liệu Sau Hợp Nhất

Sau khi hoàn tất quy trình hợp nhất và phân chia, tập dữ liệu đạt được kết quả như Bảng 3.2.

**Bảng 3.2 — Thống kê phân bố dữ liệu sau split**

| Tập dữ liệu | Tổng số ảnh | Ảnh thật (Real) | Ảnh giả (Fake) | Tỷ lệ thật |
|---|---|---|---|---|
| Huấn luyện (Train) | 221.571 | 104.819 | 116.752 | 47,3% |
| Kiểm định (Val) | 47.479 | 22.461 | 25.018 | 47,3% |
| Kiểm tra (Test) | 47.480 | 22.461 | 25.019 | 47,3% |
| **Tổng cộng** | **316.530** | **149.741** | **166.789** | **47,3%** |

Tỷ lệ phân bố nhãn được duy trì nhất quán ở mức 47,3% thật / 52,7% giả trên cả ba tập — xác nhận stratified split hoạt động chính xác. Mức mất cân bằng lớp nhẹ (~5,4 điểm phần trăm) không đủ để đòi hỏi điều chỉnh trọng số lớp trong hàm mất mát.

---

### 3.2 Quy Trình Khai Phá Dữ Liệu (EDA)

Phân tích khai phá dữ liệu (Exploratory Data Analysis — EDA) được thực hiện tại notebook `notebooks/02_eda.ipynb` vào ngày 17/05/2026, nhằm xác nhận tính toàn vẹn của pipeline tiền xử lý, hiểu đặc điểm phân bố và chất lượng ảnh, đồng thời cung cấp cơ sở thực nghiệm cho các quyết định về siêu tham số và chiến lược tăng cường dữ liệu.

#### 3.2.1 Thống Kê Mô Tả

**Phân bố lớp theo nguồn dữ liệu:**  
Hình 3.1 trình bày phân bố số lượng ảnh thật và giả trong từng bộ dữ liệu nguồn.

> **Hình 3.1** — Phân bố lớp theo nguồn dữ liệu  
> Nguồn: `reports/eda/assets/eda_class_distribution.png`

Bộ 140k-StyleGAN có phân bố hoàn toàn cân bằng (70.000 thật — 70.000 giả) và chiếm khoảng 44% tổng thể. Ưu thế quy mô của bộ này cần được lưu ý: nếu không có chiến lược lấy mẫu cân bằng theo nguồn, mô hình có nguy cơ chủ yếu học đặc trưng của ảnh StyleGAN2 thay vì học cách phân biệt thật/giả theo nghĩa tổng quát.

**Bảng 3.3 — Phân bố dữ liệu theo nguồn**

| Nguồn | Tổng | Thật | Giả | Tỷ lệ T/G |
|---|---|---|---|---|
| 140k-StyleGAN | 140.000 | 70.000 | 70.000 | 1,000 |
| Deepfake-Real | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| Hard-FakeReal | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| ciplab | [INSERT] | [INSERT] | [INSERT] | [INSERT] |
| **Tổng cộng** | **316.530** | **149.741** | **166.789** | **0,898** |

> Các ô [INSERT] sẽ được điền từ output cell 11 của `notebooks/02_eda.ipynb`.

**Phân bố độ phân giải:**  
EDA kiểm tra kích thước ảnh gốc của mẫu ngẫu nhiên 5.000 ảnh từ toàn bộ tập dữ liệu. Kết quả (Hình 3.2) cho thấy:

> **Hình 3.2** — Phân bố độ phân giải và tỷ lệ khung hình  
> Nguồn: `reports/eda/assets/eda_resolution.png`

- Ảnh 140k-StyleGAN đã ở kích thước 224×224 — phù hợp trực tiếp với đầu vào EfficientNet-B0.
- Ảnh Deepfake-Real có kích thước gốc 256×256, việc resize về 224×224 chỉ mất tối thiểu thông tin.
- Hard-FakeReal và ciplab có phân bố kích thước đa dạng hơn.
- Hầu hết ảnh có tỷ lệ khung hình xấp xỉ 1:1 (vuông) — resize về 224×224 không gây biến dạng hình học đáng kể.

Kết quả này xác nhận rằng bước resize đơn giản về 224×224 là phù hợp, không cần thêm viền (padding) hay cắt xén phức tạp.

#### 3.2.2 Phân Tích Thuộc Tính Pixel

EDA thực hiện thống kê cường độ pixel theo từng kênh màu R/G/B, lấy mẫu tối đa 3.000 ảnh mỗi nguồn, resize về 224×224 và chuẩn hóa giá trị về khoảng [0, 1].

> **Hình 3.3** — Thống kê cường độ pixel R/G/B theo nguồn  
> Nguồn: `reports/eda/assets/eda_pixel_stats.png`

Kết quả cho thấy phân bố pixel **tương đối đồng nhất** giữa bốn nguồn dữ liệu trong cùng kênh màu — không có hiện tượng lệch miền màu sắc (color domain shift) cực đoan. Kênh R có xu hướng cao hơn kênh B, phù hợp với đặc trưng tông da người. Điều này xác nhận rằng việc chuẩn hóa theo chuẩn ImageNet (mean = [0,485; 0,456; 0,406], std = [0,229; 0,224; 0,225]) là lựa chọn phù hợp cho toàn bộ tập dữ liệu, và không cần áp dụng cân bằng histogram hay bước xử lý màu bổ sung.

#### 3.2.3 Phân Tích Mẫu Đại Diện Theo Nguồn

Hình 3.4 trình bày lưới ảnh mẫu gồm 4 ảnh thật và 4 ảnh giả từ mỗi nguồn, với viền xanh lá cho ảnh thật và viền đỏ cho ảnh giả.

> **Hình 3.4** — Lưới ảnh mẫu theo nguồn và nhãn  
> Nguồn: `reports/eda/assets/eda_sample_grid.png`

Quan sát trực quan cho thấy sự khác biệt rõ rệt về đặc điểm ảnh giả giữa các nguồn:

- **140k-StyleGAN:** Ảnh giả (StyleGAN2) có chất lượng rất cao, gần như không thể phân biệt bằng mắt thường. Artifact — lỗi thị giác nhỏ mà AI để lại — chủ yếu xuất hiện ở vùng tai, hoa tai, nền ảnh và các chi tiết tóc.
- **Deepfake-Real:** Ảnh giả là khuôn mặt được ghép hoặc biến đổi (face manipulation). Vùng biên khuôn mặt và phần tiếp giáp với cổ hoặc nền thường không tự nhiên — dễ nhận ra hơn so với StyleGAN.
- **Hard-FakeReal:** Như tên gọi, bộ này được thiết kế để tối đa hóa độ khó phân biệt. Ảnh giả có độ tương phản cao, nhân vật đa dạng sắc tộc và điều kiện ánh sáng; artifact nếu có thường rất tinh vi.
- **ciplab:** Ảnh giả tổng hợp từ nhiều phương pháp GAN, chất lượng biến thiên từ dễ đến khó phân biệt.

Quan sát này xác nhận rằng mô hình cần học **đặc trưng tần số cao** (chi tiết cục bộ ở mức pixel siêu nhỏ) thay vì chỉ dựa vào màu sắc hay hình dạng tổng quát. Đây là cơ sở để ưu tiên kiến trúc CNN nhiều tầng như EfficientNet-B0 và bổ sung phân tích miền tần số (FFT).

#### 3.2.4 Phân Tích Phổ Tần Số (FFT)

Phân tích biến đổi Fourier 2D được thực hiện trên mẫu tối đa 500 ảnh mỗi lớp mỗi nguồn, chuyển về ảnh xám 224×224. Phổ log-magnitude trung bình được tính riêng cho ảnh thật và ảnh giả.

> **Hình 3.5** — Phân tích phổ tần số FFT: Thật vs. Giả  
> Nguồn: `reports/eda/assets/eda_fft.png`

Kết quả phân tích tần số tiết lộ một đặc điểm quan trọng:

- **Phổ ảnh thật:** Năng lượng tập trung ở tần số thấp (trung tâm phổ), suy giảm tự nhiên và đều đặn ra ngoài — đặc trưng của ảnh chụp thực được tạo bởi quá trình quang học tự nhiên.
- **Phổ ảnh giả:** Tương tự nhưng thường xuất hiện **năng lượng dư thừa ở mức chi tiết nhỏ** (tần số cao) — dấu vết của quá trình AI tạo ảnh, tương tự như các "nốt sai" mà phần mềm để lại.
- **Hiệu phổ (Thật − Giả):** Ảnh giả thiếu đi sự mịn màng tự nhiên ở mức chi tiết vừa và có nhiễu bất thường ở mức cực nhỏ — đây chính là "dấu vân tay kỹ thuật số" mà mô hình học để nhận diện.

Phát hiện này lý giải tại sao EfficientNet-B0 — một kiến trúc CNN học đặc trưng cục bộ ở nhiều mức độ chi tiết khác nhau — là lựa chọn phù hợp cho bài toán này. Nó cũng là cơ sở để bổ sung bước nén ảnh nhẹ (`albumentations.ImageCompression`) vào pipeline tăng cường dữ liệu, giúp mô hình nhận diện tốt cả ảnh đã qua nén (gửi qua mạng xã hội, chụp màn hình).

#### 3.2.5 Kiểm Tra Tính Toàn Vẹn Dữ Liệu (Data Leakage Check)

EDA thực hiện kiểm tra chồng lấp giữa ba tập train/val/test bằng phép giao tập hợp đường dẫn ảnh.

Kết quả:
- Train ∩ Val: không có ảnh trùng lặp
- Train ∩ Test: không có ảnh trùng lặp
- Val ∩ Test: không có ảnh trùng lặp

Ba tập hoàn toàn độc lập — xác nhận pipeline phân chia hợp lệ và kết quả đánh giá trên tập test là đáng tin cậy, không bị ảnh hưởng bởi rò rỉ dữ liệu.

Phân bố nhãn trong mỗi tập được xác nhận nhất quán ở mức 47,3% thật (xem Bảng 3.2), chứng minh stratified split hoạt động đúng.

---

### 3.3 Nhận Xét Chất Lượng Bộ Dữ Liệu

#### 3.3.1 Kích Thước và Cấu Trúc

Với tổng cộng **316.530 ảnh** sau loại bỏ trùng lặp MD5 — trong đó 221.571 ảnh dùng để huấn luyện — bộ dữ liệu có quy mô đủ lớn để huấn luyện mô hình deep learning mà không lo thiếu dữ liệu. Mức độ mất cân bằng lớp nhẹ (47,3% thật — 52,7% giả) không đủ để đòi hỏi điều chỉnh trọng số lớp: như phân tích ở mục 7.2 của tài liệu huấn luyện, Precision cao (0,9773) trên tập test bác bỏ giả thuyết mô hình thiên vị về lớp đa số.

#### 3.3.2 Chất Lượng Ảnh

Phần lớn ảnh có chất lượng tốt, ở kích thước vuông xấp xỉ 224×224 hoặc 256×256, không có ảnh panorama hay tỷ lệ khung hình bất thường. Quá trình dedup MD5 đảm bảo không có ảnh giống hệt nhau tồn tại đồng thời trong nhiều tập dữ liệu. [INSERT: số nhóm trùng lặp và tổng số ảnh đã loại từ output `notebooks/02_eda.ipynb`, Phần 7]

#### 3.3.3 Thách Thức Kỹ Thuật

**Lệch miền giữa các nguồn (domain shift):** Bốn bộ dữ liệu khác nhau về phương pháp sinh ảnh, độ phân giải gốc, và đặc điểm artifact. Mô hình huấn luyện trên hỗn hợp này có thể gặp khó khăn khi test trên ảnh từ phương pháp sinh chưa từng gặp — đây là lý do cốt lõi để thiết kế bài kiểm tra cross-generator (xem Chương 5).

**Hard-FakeReal:** Bộ dữ liệu này đặc biệt thách thức do ảnh giả được thiết kế có chủ đích để khó phân biệt, kể cả với mắt thường. Mô hình cần học đặc trưng tinh vi ở mức pixel để xử lý những trường hợp khó này.

**Ưu thế số lượng của 140k-StyleGAN (~44% tổng thể):** Nếu không cẩn thận, mô hình có thể học thiên về phong cách ảnh StyleGAN2 — phản ánh qua hiệu suất tốt trên tập test (do test set cũng chứa ảnh 140k-StyleGAN) nhưng kém hơn khi đối mặt với các bộ dữ liệu khác trong bài kiểm tra cross-generator. Vấn đề này được giải quyết một phần bởi sự hiện diện của ba bộ dữ liệu đa dạng còn lại trong tập huấn luyện.

---

## Chương 4 — Phương Pháp Đề Xuất

### 4.1 Mô Hình

[INSERT: EfficientNet-B0 backbone, custom head, 3-stage training]

### 4.2 Công Nghệ Sử Dụng

[INSERT: bảng công nghệ]

---

## Chương 5 — Thực Nghiệm

### 5.1 Quy Trình Thực Nghiệm

[INSERT]

### 5.2 Độ Đo Đánh Giá

[INSERT]

### 5.3 Thông Số Chi Tiết Các Giai Đoạn Huấn Luyện

[INSERT]

### 5.4 Đánh Giá Thực Nghiệm

[INSERT: điền sau khi có artifacts/]

### 5.5 Ứng Dụng Thực Nghiệm (Demo)

[INSERT]

---

## Chương 6 — Kết Luận

### 6.1 Tóm Tắt Đề Tài

[INSERT: 3–4 câu]

### 6.2 Đóng Góp Của Đề Tài

[INSERT]

---

## Chương 7 — Hướng Phát Triển Đề Tài

[INSERT]

---

## Tài Liệu Tham Khảo

[INSERT: ≥ 7 bài từ task_plan_ai_face_detection.md]
