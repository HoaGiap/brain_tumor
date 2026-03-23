# Hướng dẫn Tích hợp (Integration Guide)

Tài liệu này hướng dẫn cách tích hợp các module AI của dự án này vào một hệ thống khác (như `med-ai`).

## 1. Cấu trúc Module
Sau khi refactor, dự án có cấu trúc modular như sau:
- `src/models/`: Định nghĩa kiến trúc mạng CNN.
- `src/inference/engine.py`: Lớp `InferenceEngine` tập trung toàn bộ logic dự đoán.
- `src/utils/gradcam.py`: Logic tạo bản đồ nhiệt Grad-CAM.
- `src/config.py`: Cấu hình hệ thống.

## 2. Cách tích hợp vào dự án mới
Để sử dụng logic AI trong dự án mới, bạn chỉ cần thực hiện 3 bước:

### Bước 1: Copy thư mục `src`
Copy toàn bộ thư mục `src` vào thư mục gốc của dự án mới.

### Bước 2: Cài đặt thư viện
Đảm bảo dự án mới có các thư viện sau (xem `requirements.txt`):
```bash
pip install torch torchvision albumentations opencv-python pillow flask flask-cors
```

### Bước 3: Sử dụng `InferenceEngine`
Đây là cách gọi đơn giản nhất để thực hiện dự đoán từ mã Python của bạn:

```python
from PIL import Image
from src.inference.engine import InferenceEngine

# 1. Khởi tạo Engine
engine = InferenceEngine()

# 2. Load các model mong muốn (ResNet, EfficientNet...)
engine.load_model("resnet50", "path/to/resnet50_best.pth")
engine.load_model("efficientnet", "path/to/effnet_best.pth")

# 3. Chạy dự đoán từ ảnh PIL
img = Image.open("your_mri_scan.jpg")
result = engine.predict(img, gradcam_request="both")

# 4. Sử dụng kết quả
print(f"Kết quả: {result['class_vi']} ({result['confidence']}%)")
# result['gradcam'] chứa ảnh base64 để hiển thị lên UI
```

## 3. API Endpoints (Nếu dùng Flask)
Nếu bạn muốn dùng API, bạn có thể tham khảo `app.py`. Endpoint quan trọng nhất là:
- **POST `/predict`**: Nhận file ảnh và trả về JSON kết quả gồm phân loại, xác suất từng lớp và ảnh Grad-CAM.

---
**Lưu ý**: Đảm bảo GPU (CUDA) đã được cài đặt để đạt hiệu năng tốt nhất. Kiểm tra `DEVICE` trong `src/config.py`.
