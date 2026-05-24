# Model Usage Guide - AI-Generated Face Detection

**Model đã train xong với kết quả:**
- Test Accuracy: **97.33%**
- Test F1-Score: **97.45%**
- Test AUC-ROC: **99.70%**

---

## 1. Model Location

Model đã train nằm tại:
```
artifacts/checkpoints/best_stage3.pth
```

Đây là checkpoint tốt nhất sau khi train 3 stages (freeze backbone → unfreeze top blocks → unfreeze all).

---

## 2. Predict cho 1 ảnh

### Cú pháp

```bash
python scripts/predict.py <đường_dẫn_ảnh>
```

### Ví dụ

```bash
# Predict ảnh từ test set
python scripts/predict.py "data/raw/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test/real/00001.jpg"

# Predict ảnh từ máy tính
python scripts/predict.py "C:/Users/Admin/Downloads/face.jpg"
```

### Output

```
Device: cuda
Loaded checkpoint: artifacts/checkpoints/best_stage3.pth
Checkpoint epoch: 30
Val accuracy: 0.9733

Processing: path/to/image.jpg

==================================================
Prediction: Real
Confidence: 98.45%
==================================================
```

---

## 3. Đánh giá lại model trên test set

Nếu muốn tính lại metrics (accuracy, precision, recall, F1, AUC-ROC) và vẽ lại biểu đồ:

```bash
python scripts/run_evaluate.py
```

Hoặc chỉ định checkpoint cụ thể:

```bash
python scripts/run_evaluate.py --checkpoint artifacts/checkpoints/best_stage3.pth
```

Kết quả sẽ được lưu vào:
- `artifacts/results/metrics.txt` - Các chỉ số đánh giá
- `artifacts/results/confusion_matrix.png` - Ma trận nhầm lẫn
- `artifacts/results/roc_curve.png` - Đường cong ROC

---

## 4. Sử dụng model trong Python code

### Load model

```python
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append("src")
from model import FaceDetectionModel

# Create model
model = FaceDetectionModel(
    backbone_name="efficientnet_b0",
    pretrained=False,
    hidden_dim=256,
    dropout=0.5
)

# Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("artifacts/checkpoints/best_stage3.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("Model loaded successfully!")
```

### Preprocess ảnh

```python
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def preprocess_image(image_path):
    # Load image
    with Image.open(image_path) as img:
        image = np.array(img.convert("RGB"))
    
    # Transform (resize + normalize)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed["image"]
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 224, 224)
    
    return image_tensor
```

### Predict

```python
def predict(model, image_path, device):
    # Preprocess
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()
    
    # Interpret result
    # prob > 0.5 → Fake
    # prob <= 0.5 → Real
    label = "Fake" if prob > 0.5 else "Real"
    confidence = prob if prob > 0.5 else (1 - prob)
    
    return label, confidence

# Example usage
label, confidence = predict(model, "path/to/image.jpg", device)
print(f"Prediction: {label}")
print(f"Confidence: {confidence*100:.2f}%")
```

---

## 5. Batch Prediction (nhiều ảnh cùng lúc)

```python
import torch
from torch.utils.data import DataLoader
import pandas as pd

def predict_batch(model, image_paths, device, batch_size=32):
    """
    Predict cho nhiều ảnh cùng lúc
    
    Args:
        model: Trained model
        image_paths: List of image paths
        device: torch.device
        batch_size: Batch size for inference
    
    Returns:
        results: List of (image_path, label, confidence)
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                tensor = preprocess_image(path)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"[ERROR] Failed to load {path}: {e}")
                results.append((path, "Error", 0.0))
                continue
        
        if len(batch_tensors) == 0:
            continue
        
        # Stack into batch
        batch = torch.cat(batch_tensors, dim=0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # Interpret results
        for path, prob in zip(batch_paths, probs):
            label = "Fake" if prob > 0.5 else "Real"
            confidence = prob if prob > 0.5 else (1 - prob)
            results.append((path, label, confidence))
    
    return results

# Example usage
image_paths = [
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    "path/to/image3.jpg"
]

results = predict_batch(model, image_paths, device, batch_size=32)

# Save to CSV
df = pd.DataFrame(results, columns=["image_path", "prediction", "confidence"])
df.to_csv("predictions.csv", index=False)
print(f"Saved predictions for {len(results)} images")
```

---

## 6. Model Information

### Architecture
- **Backbone:** EfficientNet-B0 (pretrained on ImageNet)
- **Classifier Head:** Linear(1280 → 256) → ReLU → Dropout(0.5) → Linear(256 → 1)
- **Total Parameters:** 4,335,741
- **Input Size:** 224×224 RGB images
- **Output:** Single logit (use sigmoid to get probability)

### Training Details
- **3-stage progressive unfreezing:**
  - Stage 1: Freeze backbone, train head only (LR=1e-4)
  - Stage 2: Unfreeze top 2 blocks (LR=1e-5)
  - Stage 3: Unfreeze all (LR=1e-6)
- **Optimizer:** AdamW (weight_decay=1e-4)
- **Loss:** BCEWithLogitsLoss
- **Mixed Precision:** Enabled (FP16)
- **Training Time:** ~6-8 hours on RTX 5070

### Dataset
- **Total Images:** 316,530 (after deduplication)
- **Train:** 221,571 images (70%)
- **Val:** 47,479 images (15%)
- **Test:** 47,480 images (15%)
- **Sources:** 4 datasets (140k StyleGAN, Deepfake-Real, Hard-FakeReal, ciplab)

---

## 7. Interpretation Guide

### Confidence Levels

| Confidence | Interpretation |
|------------|----------------|
| 95-100% | Very confident - model is almost certain |
| 85-95% | Confident - reliable prediction |
| 70-85% | Moderate confidence - likely correct but verify if critical |
| 50-70% | Low confidence - borderline case, manual review recommended |

### Label Meaning

- **Real:** Model predicts this is a real human face (not AI-generated)
- **Fake:** Model predicts this is an AI-generated face (StyleGAN, deepfake, etc.)

### Probability Interpretation

Model outputs a probability `p`:
- `p > 0.5` → Fake (confidence = p)
- `p <= 0.5` → Real (confidence = 1 - p)

Example:
- `p = 0.98` → Fake with 98% confidence
- `p = 0.15` → Real with 85% confidence (1 - 0.15)

---

## 8. Limitations

Model được train trên 4 datasets cụ thể, nên có thể gặp hạn chế:

1. **Generator-specific:** Model học patterns từ StyleGAN, manipulation-based fakes. Có thể kém chính xác với generators mới (DALL-E, Midjourney, Stable Diffusion).

2. **Robustness:** Performance có thể giảm khi ảnh bị:
   - JPEG compression mạnh
   - Resize nhiều lần
   - Blur/noise

3. **Domain shift:** Model train trên ảnh khuôn mặt. Không dùng cho:
   - Ảnh toàn thân
   - Ảnh phong cảnh
   - Ảnh không có khuôn mặt

4. **Confidence calibration:** Confidence score chưa được calibrated - không phản ánh chính xác xác suất thực tế.

---

## 9. Next Steps

Để cải thiện model hoặc mở rộng project:

1. **Cross-Generator Testing** (Task plan section 6)
   - Test model trên generators chưa thấy
   - Đánh giá khả năng generalize

2. **Robustness Testing** (Task plan section 7)
   - Test với JPEG compression, blur, resize
   - Đo accuracy drop

3. **Grad-CAM Explainability** (Task plan section 8)
   - Visualize model "nhìn" vào đâu
   - Hiểu cách model ra quyết định

4. **Demo Webapp** (Task plan section 9)
   - Tạo Gradio interface
   - Upload ảnh → Real/Fake + confidence + Grad-CAM heatmap

---

## 10. Troubleshooting

### Lỗi: "CUDA out of memory"
→ Giảm batch_size trong batch prediction

### Lỗi: "No module named 'src'"
→ Chạy từ root directory của project, không phải từ `scripts/`

### Lỗi: "FileNotFoundError: checkpoint not found"
→ Kiểm tra đường dẫn checkpoint, phải chạy từ root directory

### Model predict sai
→ Kiểm tra:
- Ảnh có phải khuôn mặt không?
- Ảnh có bị corrupt không?
- Ảnh có đúng format (RGB) không?

---

## 11. Contact & References

- **Task Plan:** `task_plan_ai_face_detection.md`
- **Training Guide:** `TRAINING_GUIDE.md`
- **Hyperparameters:** `HYPERPARAMETERS.md`
- **Config:** `configs/train_config.yaml`

**Model Performance:**
- Test Accuracy: 97.33%
- Test F1-Score: 97.45%
- Test AUC-ROC: 99.70%

**Training Date:** 2026-05-20
