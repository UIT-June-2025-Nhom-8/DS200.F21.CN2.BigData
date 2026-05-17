# DS200.F21.CN2 — Báo Cáo Phân Tích Khám Phá Dữ Liệu (EDA)

> **Notebook thực thi:** `notebooks/02_eda.ipynb`  
> **Hình ảnh đầu ra:** `reports/eda/assets/`  
> **Ngày thực hiện:** 17/05/2026

---

## Tổng Quan

Phân tích khám phá dữ liệu (Exploratory Data Analysis — EDA) được thực hiện trên tập dữ liệu tổng hợp từ 4 nguồn Kaggle, sau khi đã loại bỏ trùng lặp MD5 và chia tập 70/15/15 tại notebook `01_dataset_preparation.ipynb`. Mục tiêu của EDA là:

1. Xác nhận tính toàn vẹn của pipeline tiền xử lý.
2. Hiểu sâu đặc điểm phân bố, chất lượng và nội dung ảnh.
3. Cung cấp bằng chứng thực nghiệm để lựa chọn siêu tham số, chiến lược tăng cường dữ liệu và kiến trúc mô hình.

---

## Phần 1 — Phân Bố Lớp Theo Nguồn Dữ Liệu

**Hình:** `reports/eda/assets/eda_class_distribution.png`

### Mục tiêu

Kiểm tra xem số ảnh Thật và Giả trong từng nguồn có gần bằng nhau không. Nếu quá lệch, mô hình AI sẽ bị "thiên vị" — xu hướng đoán theo lớp nhiều hơn mà không thực sự học cách phân biệt.

### Phương pháp

- Tổng hợp số lượng ảnh theo cặp `(nguồn, nhãn)`.
- Vẽ biểu đồ cột số lượng tuyệt đối (trái) và biểu đồ tỷ lệ phần trăm (phải).

### Kết quả quan sát

- **140k-StyleGAN:** cân bằng hoàn hảo — 70.000 ảnh Thật (Flickr) và 70.000 ảnh Giả (StyleGAN2), chiếm phần lớn tổng thể.
- **Deepfake-Real:** [INSERT: số lượng Thật / Giả từ output notebook].
- **Hard-FakeReal:** tập nhỏ, [INSERT: số lượng]; tỷ lệ [INSERT: %] Thật.
- **ciplab:** [INSERT: số lượng Thật / Giả].
- Tổng thể: [INSERT: tổng Real] Thật | [INSERT: tổng Fake] Giả — tỷ lệ [INSERT: %] cân bằng / mất cân bằng.

### Ứng dụng cho bước tiếp theo

| Phát hiện                            | Hành động                                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Nếu tổng tỷ lệ Thật/Giả lệch > 10 pp | Cân bằng trọng số huấn luyện để mô hình chú ý đều cả hai lớp (tham số `pos_weight` trong hàm mất mát)                      |
| 140k-StyleGAN chiếm ~90% tổng số     | Lấy mẫu cân bằng theo nguồn khi huấn luyện để tránh mô hình chỉ học phong cách ảnh StyleGAN (lấy mẫu phân tầng theo nguồn) |
| Nguồn ciplab / Hard-FakeReal nhỏ     | Áp dụng tăng cường dữ liệu (augmentation) mạnh hơn cho hai nguồn này để mô hình học được đa dạng hơn                       |

---

## Phần 2 — Chất Lượng Ảnh: Độ Phân Giải & Tỷ Lệ Khung Hình

**Hình:** `reports/eda/assets/eda_resolution.png`

### Mục tiêu

Xác định khoảng phân giải và tỷ lệ khung hình thực tế của dữ liệu thô để thiết kế bước resize phù hợp, tránh mất thông tin không cần thiết hoặc biến dạng ảnh.

### Phương pháp

- Lấy mẫu ngẫu nhiên tối đa 5.000 ảnh, đọc kích thước gốc bằng Pillow.
- Biểu đồ scatter `(width, height)` với đường tham chiếu 224×224.
- Histogram tỷ lệ khung hình `width / height`.

### Kết quả quan sát

- Phần lớn ảnh **140k-StyleGAN** đã ở 224×224 — phù hợp trực tiếp với đầu vào EfficientNet-B0.
- Ảnh **Deepfake-Real** có kích thước gốc 256×256, resize về 224×224 mất tối thiểu thông tin.
- **Hard-FakeReal** và **ciplab** có phân bố phân giải đa dạng hơn.
- Hầu hết ảnh có tỷ lệ khung hình xấp xỉ 1:1 (vuông) — resize không gây biến dạng đáng kể.
- [INSERT: % ảnh đã ở đúng 224×224 từ dòng in ra của notebook].

### Ứng dụng cho bước tiếp theo

- **Pipeline tiền xử lý xác nhận:** Resize 224×224 là hợp lý, không cần padding.
- Không phát hiện ảnh dạng panorama hay tỷ lệ bất thường — bước `albumentations.Resize(224, 224)` đủ dùng mà không cần `LongestMaxSize` + padding.
- Nếu có ảnh rất nhỏ (< 64 px), cân nhắc thêm bộ lọc loại bỏ trước khi train.

---

## Phần 3 — Lưới Ảnh Mẫu: Thật vs Giả Theo Từng Nguồn

**Hình:** `reports/eda/assets/eda_sample_grid.png`

### Mục tiêu

Quan sát trực quan sự khác biệt giữa ảnh Thật và Giả theo từng nguồn. Phát hiện các dấu hiệu giả tạo có thể nhìn thấy bằng mắt thường — lỗi ở vùng tóc/tai (gọi là "artifact"), màu da bất thường, răng, hoặc nền ảnh không tự nhiên.

### Phương pháp

- Lấy ngẫu nhiên 4 ảnh Thật và 4 ảnh Giả cho mỗi nguồn (8 cột × 4 hàng).
- Viền **xanh lá** cho ảnh Thật, viền **đỏ** cho ảnh Giả.

### Kết quả quan sát

- **140k-StyleGAN:** Ảnh giả (StyleGAN2) rất thuyết phục, khó phân biệt bằng mắt thường; đôi khi có artifact ở vùng tai, hoa tai hoặc nền ảnh.
- **Deepfake-Real:** Ảnh giả thường là khuôn mặt được ghép/biến đổi — dễ nhận ra hơn StyleGAN nhờ vùng biên khuôn mặt không tự nhiên.
- **Hard-FakeReal:** Như tên gọi, ảnh giả thiết kế để khó phân biệt — độ tương phản cao, nhân vật đa dạng sắc tộc.
- **ciplab:** Ảnh giả tổng hợp từ nhiều phương pháp GAN; chất lượng biến thiên.

### Ứng dụng cho bước tiếp theo

- Xác nhận rằng mô hình cần học **đặc trưng tần số cao** (artifact), không chỉ màu sắc hay hình dạng tổng quát → phù hợp với phân tích FFT ở Phần 5.
- Các ảnh **ciplab** và **Hard-FakeReal** nên được đại diện đầy đủ trong tập test để đánh giá khả năng tổng quát hóa.
- Ghi nhận để viết phần **Grad-CAM**: quan sát mô hình tập trung vào vùng nào (mắt, mũi, tóc, hay nền ảnh).

---

## Phần 4 — Thống Kê Cường Độ Pixel R/G/B Theo Nguồn

**Hình:** `reports/eda/assets/eda_pixel_stats.png`

### Mục tiêu

Kiểm tra xem màu sắc và độ sáng trung bình của ảnh từ các nguồn khác nhau có tương đồng không. Nếu một nguồn có ảnh quá tối hay quá sáng bất thường so với các nguồn còn lại (**domain shift** — lệch miền dữ liệu), mô hình có thể học "lối tắt" theo màu sắc thay vì học cách phát hiện ảnh giả thực sự.

### Phương pháp

- Lấy mẫu tối đa 3.000 ảnh mỗi nguồn.
- Tính giá trị trung bình pixel từng kênh R/G/B sau khi resize về 224×224, chuẩn hóa về [0, 1].
- Biểu đồ hộp (box plot) cho từng kênh theo nguồn.

### Kết quả quan sát

- Phân bố pixel **tương đối đồng nhất** giữa các nguồn trong cùng kênh màu — không có domain shift màu sắc cực đoan.
- Kênh R có xu hướng cao hơn kênh B — phù hợp với ảnh khuôn mặt người (tông da).
- [INSERT: quan sát cụ thể từ biểu đồ — nguồn nào có median cao/thấp bất thường].

### Ứng dụng cho bước tiếp theo

- **Chuẩn hóa theo chuẩn ImageNet** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) là phù hợp — phân bố pixel không lệch xa chuẩn ImageNet (bộ ảnh lớn mà mô hình đã học trước).
- Nếu phát hiện nguồn có màu lệch mạnh, thêm `albumentations.ColorJitter` vào bước tăng cường dữ liệu để mô hình bất biến với màu sắc.
- Dùng làm bằng chứng để **không cần** xử lý cân bằng histogram (histogram equalization / CLAHE) — tập dữ liệu đã đủ đồng nhất.

---

## Phần 5 — Phân Tích Phổ Tần Số (FFT): Thật vs Giả

**Hình:** `reports/eda/assets/eda_fft.png`

### Mục tiêu

Kiểm tra xem ảnh giả có để lại **"dấu vân tay kỹ thuật số"** ẩn hay không — ảnh do AI tạo thường có các rung động bất thường ở mức chi tiết siêu nhỏ mà mắt thường không thấy. Đây là cơ sở để chọn kiến trúc mô hình và chiến lược tăng cường dữ liệu phù hợp.

### Phương pháp

- Lấy mẫu tối đa 500 ảnh/lớp/nguồn, chuyển về ảnh xám 224×224.
- Tính biến đổi Fourier 2D (`np.fft.fft2`), log-scale magnitude, dịch tâm về giữa (`fftshift`).
- Tính phổ trung bình cho Thật và Giả riêng biệt; tính hiệu (Thật − Giả).

### Kết quả quan sát

- **Phổ Thật:** Năng lượng tập trung ở tần số thấp (trung tâm), suy giảm tự nhiên ra ngoài — đặc trưng của ảnh chụp thực.
- **Phổ Giả:** Tương tự nhưng thường có **năng lượng dư thừa ở mức chi tiết nhỏ** — dấu vết của quá trình AI tạo ảnh, giống như các "nốt sai" mà phần mềm để lại sau khi tạo ra hình ảnh.
- **Hiệu phổ (Thật − Giả):** Ảnh giả thiếu đi sự mịn màng tự nhiên ở mức chi tiết vừa, thay vào đó có nhiễu lộn xộn ở mức cực nhỏ — đây chính là "dấu vân tay" mà mô hình học để nhận diện.
- [INSERT: mô tả cụ thể pattern quan sát được từ hình `eda_fft.png`].

### Ứng dụng cho bước tiếp theo

- Kết quả này lý giải vì sao dùng **EfficientNet-B0** (mô hình CNN học đặc trưng cục bộ ở nhiều mức độ chi tiết khác nhau) là lựa chọn phù hợp.
- Nếu sau huấn luyện ban đầu mô hình vẫn chưa đạt yêu cầu, xem xét bổ sung thêm nhánh phân tích tần số song song.
- Đưa hình `eda_fft.png` vào phần **Methodology** của báo cáo để minh họa đặc điểm phân biệt Thật/Giả ở miền tần số.
- Ưu tiên thêm bước **nén ảnh nhẹ** (`albumentations.ImageCompression`) khi tăng cường dữ liệu — giúp mô hình nhận diện tốt cả ảnh bị giảm chất lượng (gửi qua mạng xã hội, chụp màn hình).

---

## Phần 6 — Tính Toàn Vẹn: Kiểm Tra Chồng Lấp Train/Val/Test

**Đầu ra:** In trực tiếp ra màn hình (không có file hình ảnh).

### Mục tiêu

Đảm bảo quá trình chia dữ liệu không bị "gian lận" (**data leakage** — rò rỉ dữ liệu) — tức là không có ảnh nào xuất hiện đồng thời trong nhiều hơn một tập (dạy và kiểm tra cùng ảnh).

### Phương pháp

- So sánh tập hợp `image_path` giữa train, val, test bằng phép toán giao (`∩`).
- In kết quả kiểm tra từng cặp: Train∩Val, Train∩Test, Val∩Test.
- In bảng phân bố nhãn theo từng tập để xác nhận stratified split hoạt động đúng.

### Kết quả quan sát

- ✓ **Train ∩ Val:** Không trùng.
- ✓ **Train ∩ Test:** Không trùng.
- ✓ **Val ∩ Test:** Không trùng.
- Ba tập hoàn toàn độc lập — pipeline split hợp lệ.
- Tỷ lệ nhãn trong train/val/test xấp xỉ nhau (stratified split thành công): [INSERT: % Thật trong mỗi tập từ bảng notebook].

### Ứng dụng cho bước tiếp theo

- **Xác nhận:** Kết quả đánh giá trên test set là đáng tin cậy — không bị rò rỉ từ train.
- Nếu phát hiện overlap, phải chạy lại `01_dataset_preparation.ipynb` với `random_state=42` và kiểm tra logic `train_test_split`.
- Bảng phân bố nhãn đặt vào phần **Dataset** của báo cáo chính thức để chứng minh tính nhất quán.

---

## Phần 7 — MD5 Duplicate Viewer

**Hình:** `reports/eda/assets/eda_duplicate_viewer.png` (chỉ tạo ra nếu có trùng lặp)  
**Cache:** `reports/eda/assets/duplicate_hashes.parquet`

### Mục tiêu

Phát hiện ảnh trùng lặp hoàn toàn **trong dữ liệu thô** dựa trên mã MD5 (mã "vân tay" kỹ thuật số — hai file giống nhau byte-by-byte sẽ có cùng mã). Ảnh trùng giữa các nguồn khác nhau hoặc cùng ảnh được gán nhãn khác nhau là vấn đề nghiêm trọng.

### Phương pháp

- Đọc toàn bộ ảnh từ `data/raw/` (4 nguồn), tính MD5 từng file.
- Nhóm các ảnh có cùng MD5 — nhóm có > 1 thành viên là trùng lặp.
- Kết quả cache vào `duplicate_hashes.parquet` để tái sử dụng nhanh.
- Hiển thị lưới ảnh ví dụ (tối đa 6 nhóm × 3 ảnh) nếu phát hiện trùng.

### Kết quả quan sát

- [INSERT: số nhóm trùng lặp từ output notebook].
- [INSERT: tổng số ảnh trùng và số ảnh đã loại bỏ].
- [INSERT: mô tả — trùng lặp chủ yếu giữa nguồn nào với nguồn nào, hoặc ✓ không có trùng lặp].

> **Lưu ý:** Notebook `01_dataset_preparation.ipynb` đã thực hiện dedup MD5 **trước** khi tạo file CSV splits. Do đó, dữ liệu trong `data/splits/*.csv` đã sạch. Phần 7 này cung cấp thêm bằng chứng kiểm chứng độc lập.

### Ứng dụng cho bước tiếp theo

- Nếu phát hiện ảnh trùng nhãn mâu thuẫn (cùng ảnh nhưng một nguồn gán Thật, nguồn kia gán Giả), cần điều tra thủ công và loại bỏ.
- File `duplicate_hashes.parquet` có thể tái sử dụng trong `01_dataset_preparation.ipynb` để tăng tốc dedup khi thêm dataset mới.
- Kết quả này cần được nêu trong phần **Dataset** báo cáo: "Sau khi loại bỏ [INSERT: N] ảnh trùng lặp, tập dữ liệu cuối gồm [INSERT: N] ảnh."

---

## Phần 8 — Bảng Tổng Hợp Thống Kê Theo Nguồn

**Đầu ra:** In trực tiếp ra màn hình.

### Mục tiêu

Cung cấp bảng thống kê tổng hợp cuối cùng để đưa vào báo cáo và làm cơ sở quyết định cuối cùng trước khi bắt đầu huấn luyện.

### Kết quả quan sát

| Nguồn         | Tổng     | Thật     | Giả      | T/G Ratio |
| ------------- | -------- | -------- | -------- | --------- |
| 140k-StyleGAN | [INSERT] | [INSERT] | [INSERT] | [INSERT]  |
| Deepfake-Real | [INSERT] | [INSERT] | [INSERT] | [INSERT]  |
| Hard-FakeReal | [INSERT] | [INSERT] | [INSERT] | [INSERT]  |
| ciplab        | [INSERT] | [INSERT] | [INSERT] | [INSERT]  |
| **Tổng cộng** | [INSERT] | [INSERT] | [INSERT] | [INSERT]  |

> Điền số liệu thực tế từ output của cell 11 trong `notebooks/02_eda.ipynb`.

### Ứng dụng cho bước tiếp theo

- **Nếu T/G Ratio ≈ 1.000:** Không cần `pos_weight` trong loss function.
- **Nếu T/G Ratio lệch > 10%:** Đặt `pos_weight = n_fake / n_real` khi khởi tạo `BCEWithLogitsLoss`.
- Bảng này đi vào phần **4.1 Dataset** của báo cáo chính thức, thay thế các `[INSERT]`.

---

## Tóm Tắt Kết Quả EDA & Quyết Định Cho Bước Tiếp Theo

| Phát hiện từ EDA                             | Quyết định kỹ thuật                                                  |
| -------------------------------------------- | -------------------------------------------------------------------- |
| 140k-StyleGAN chiếm đa số                    | Lấy mẫu cân bằng theo nguồn khi huấn luyện (tránh thiên vị StyleGAN) |
| Ảnh hầu hết đã là 224×224 hoặc 256×256 vuông | Thu nhỏ về 224×224, không cần thêm viền hay cắt xén thêm             |
| Phân bố pixel gần với ImageNet               | Dùng công thức chuẩn hóa ImageNet trực tiếp                          |
| FFT cho thấy ảnh Giả có dấu vân tay ẩn       | Thêm bước nén ảnh nhẹ khi huấn luyện (`ImageCompression`)            |
| Ba tập train/val/test độc lập hoàn toàn      | Kết quả đánh giá cuối cùng là đáng tin cậy                           |
| [INSERT: kết quả dedup] ảnh trùng đã loại    | Ghi nhận trong phần Dataset của báo cáo                              |
| [INSERT: cân bằng/mất cân bằng lớp]          | Bật/tắt cân bằng trọng số lớp (`pos_weight`) tương ứng               |

---

## Các File Hình Ảnh Được Tạo Ra

| File                                                 | Dùng trong báo cáo tại                             |
| ---------------------------------------------------- | -------------------------------------------------- |
| `reports/eda/assets/eda_class_distribution.png`      | Phần 4.1 Dataset — Hình X: Phân bố lớp             |
| `reports/eda/assets/eda_per_source_distribution.png` | Phần 4.1 Dataset — Hình X: Phân bố theo nguồn      |
| `reports/eda/assets/eda_resolution.png`              | Phần 4.2 Tiền xử lý — Hình X: Phân bố độ phân giải |
| `reports/eda/assets/eda_resolution_scatter.png`      | Phần 4.2 Tiền xử lý — Hình X: Scatter độ phân giải |
| `reports/eda/assets/eda_sample_grid.png`             | Phần 4.1 Dataset — Hình X: Ảnh mẫu theo nguồn      |
| `reports/eda/assets/eda_pixel_stats.png`             | Phụ lục hoặc Phần 4.2                              |
| `reports/eda/assets/eda_intensity_per_source.png`    | Phụ lục hoặc Phần 4.2                              |
| `reports/eda/assets/eda_fft.png`                     | Phần 4.3 Đặc trưng — Hình X: Phân tích FFT         |
| `reports/eda/assets/eda_duplicate_viewer.png`        | Phụ lục — Minh họa ảnh trùng lặp (nếu có)          |
