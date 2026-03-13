# 🧠 NeuroScan AI — Brain Tumor MRI Classifier

Hệ thống phân loại khối u não trên ảnh MRI sử dụng **ResNet50** và **EfficientNet-B0**
với trực quan hóa **Grad-CAM++**.

---

## 📁 Cấu trúc dự án

```
brain_tumor/
├── src/
│   ├── dataset.py        # Data loading, augmentation (Albumentations)
│   └── models.py         # ResNet50, EfficientNet-B0, Grad-CAM, Grad-CAM++
├── templates/
│   └── index.html        # Giao diện HTML+JS
├── checkpoints/          # Checkpoint sau training
├── logs/                 # TensorBoard logs
├── train.py              # Script huấn luyện
├── app.py                # Flask API backend
└── requirements.txt
```

---

## ⚙️ Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 2. Cài dependencies
pip install -r requirements.txt
```

---

## 📦 Dữ liệu

Tải từ Kaggle: **Brain Tumor MRI Dataset**
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Sau khi tải về, giải nén thành cấu trúc:

```
data/
├── Training/
│   ├── glioma/          (~1321 ảnh)
│   ├── meningioma/      (~1339 ảnh)
│   ├── notumor/         (~1595 ảnh)
│   └── pituitary/       (~1457 ảnh)
└── Testing/
    ├── glioma/          (~300 ảnh)
    ├── meningioma/      (~306 ảnh)
    ├── notumor/         (~405 ảnh)
    └── pituitary/       (~300 ảnh)
```

---

## 🚀 Huấn luyện

### Huấn luyện ResNet50
```bash
python train.py \
  --backbone resnet50 \
  --data ./data \
  --epochs 50 \
  --batch 32 \
  --lr 1e-4 \
  --mode finetune
```

### Huấn luyện EfficientNet-B0
```bash
python train.py \
  --backbone efficientnet \
  --data ./data \
  --epochs 50 \
  --batch 32 \
  --lr 1e-4
```

### Huấn luyện cả hai cùng lúc
```bash
python train.py --backbone both --data ./data --epochs 50
```

### Các tham số quan trọng

| Tham số | Mặc định | Mô tả |
|---------|---------|-------|
| `--backbone` | `both` | `resnet50` / `efficientnet` / `both` |
| `--data` | `./data` | Thư mục dữ liệu |
| `--epochs` | `50` | Số epoch |
| `--batch` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--mode` | `finetune` | `finetune` / `feature` / `partial` |
| `--scheduler` | `cosine` | `cosine` / `onecycle` |
| `--img-size` | `224` | Kích thước ảnh input |
| `--no-amp` | False | Tắt mixed precision |

---

## 🌐 Chạy API Server

```bash
python app.py \
  --resnet  checkpoints/resnet50/NGÀY/resnet50_best.pth \
  --effnet  checkpoints/efficientnet/NGÀY/efficientnet_best.pth \
  --port    5000
```

Server sẽ khởi động tại: `http://localhost:5000`

### API Endpoints

| Method | Endpoint | Mô tả |
|--------|---------|-------|
| GET | `/health` | Kiểm tra trạng thái server |
| POST | `/predict` | Phân tích ảnh MRI |

### Ví dụ gọi API

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/mri.jpg" \
  -o result.json
```

### Cấu trúc response

```json
{
  "prediction": {
    "class": "glioma",
    "class_vi": "U thần kinh đệm",
    "confidence": 94.2,
    "has_tumor": true
  },
  "probabilities": {
    "glioma":     {"score_pct": 94.2, "label_vi": "U thần kinh đệm"},
    "meningioma": {"score_pct": 3.1,  "label_vi": "U màng não"},
    "notumor":    {"score_pct": 1.9,  "label_vi": "Không có u"},
    "pituitary":  {"score_pct": 0.8,  "label_vi": "U tuyến yên"}
  },
  "per_model": {
    "resnet50": {"glioma": 93.1, ...},
    "efficientnet": {"glioma": 95.3, ...}
  },
  "gradcam": {
    "resnet50": "data:image/png;base64,...",
    "efficientnet": "data:image/png;base64,..."
  },
  "severity": {"level": "high", "label": "Cao"},
  "recommendation": "..."
}
```

---

## 🖥️ Mở giao diện Web

1. Chạy Flask server (xem phần trên)
2. Mở file `templates/index.html` trong trình duyệt
3. Upload ảnh MRI → Nhấn **Phân tích MRI**

---

## 📈 Theo dõi training với TensorBoard

```bash
tensorboard --logdir ./checkpoints
```

Mở trình duyệt: `http://localhost:6006`

---

## 🏗️ Kiến trúc chi tiết

### ResNet50 (Fine-tuned)
```
ImageNet pretrained ResNet50
  └── Feature Extractor (conv1 → layer4)  [Frozen / Fine-tuned]
  └── AdaptiveAvgPool2d
  └── Custom Classifier:
        Dropout(0.4) → Linear(2048→512) → BN → ReLU
        → Dropout(0.2) → Linear(512→4)
Grad-CAM target: layer4 (last residual block)
```

### EfficientNet-B0 (Fine-tuned)
```
ImageNet pretrained EfficientNet-B0
  └── Feature Extraction (MBConv blocks)  [Frozen / Fine-tuned]
  └── AdaptiveAvgPool
  └── Custom Classifier:
        Dropout(0.4) → Linear(1280→256) → BN → SiLU
        → Dropout(0.2) → Linear(256→4)
Grad-CAM target: _blocks[-1] (last MBConv block)
```

### Ensemble
```
P_ensemble = 0.5 × softmax(ResNet50) + 0.5 × softmax(EfficientNet)
```

---

## 🔬 Kỹ thuật sử dụng

| Kỹ thuật | Mô tả |
|----------|-------|
| **Transfer Learning** | Pretrained ImageNet weights |
| **Label Smoothing** | ε=0.1 chống overfitting |
| **Weighted Sampler** | Xử lý class imbalance |
| **Albumentations** | Data augmentation mạnh |
| **Cosine LR Decay** | CosineAnnealingLR |
| **AMP** | Mixed precision training (fp16) |
| **Grad-CAM++** | Visualize vùng AI quan sát |
| **Early Stopping** | Patience=10 |
| **Ensemble** | ResNet50 + EfficientNet-B0 |

---

## 📊 Kết quả mong đợi

| Mô hình | Val Accuracy | Test Accuracy |
|---------|-------------|--------------|
| ResNet50 | ~95–97% | ~94–96% |
| EfficientNet-B0 | ~95–97% | ~94–97% |
| **Ensemble** | **~96–98%** | **~96–98%** |

*Kết quả thực tế phụ thuộc vào hyperparameters và hardware.*

---

## ⚠️ Lưu ý quan trọng

> Đây là **công cụ nghiên cứu và học thuật**.
> **KHÔNG** sử dụng để thay thế chẩn đoán y tế thực tế.
> Mọi kết quả cần được xác nhận bởi bác sĩ thần kinh chuyên khoa.

---

*Built with PyTorch · Flask · Albumentations · Grad-CAM++*
