# 🧠 NeuroScan AI — Brain Tumor MRI Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

**NeuroScan AI** là hệ thống hỗ trợ chẩn đoán khối u não từ ảnh MRI thông qua trí tuệ nhân tạo. Ứng dụng kết hợp sức mạnh của 4 kiến trúc mạng nơ-ron tích chập (CNN) tiên tiến để đưa ra kết quả phân loại ensemble chính xác cao cùng với công nghệ **Grad-CAM++** để trực quan hóa vùng quan tâm.

---

## ✨ Tính năng nổi bật

- **⚡ Quad-Model Ensemble**: Kết hợp dự đoán từ 4 mô hình: **ResNet50**, **EfficientNet-B0**, **ConvNeXt-Small**, và **EfficientNet-V2-S**.
- **🔍 Giải thích AI (XAI)**: Sử dụng **Grad-CAM++** để hiển thị heatmap, giúp bác sĩ biết AI đang tập trung vào vùng nào trên ảnh MRI.
- **🎨 Giao diện Hiện đại**: Web-app trực quan, hỗ trợ kéo thả ảnh, hiệu ứng quét (scanning) và biểu đồ so sánh mô hình theo thời gian thực.
- **🚀 Hiệu suất tối ưu**: Hỗ trợ tăng tốc phần cứng (CUDA/GPU) và xử lý ảnh chuyên sâu với Albumentations.

---

## 📁 Cấu trúc thư mục

```text
brain_tumor/
├── src/
│   ├── config.py         # ⚙️ Cấu hình tập trung (IMG_SIZE, CLASS, Port...)
│   ├── dataset.py        # Xử lý dữ liệu & Augmentation
│   ├── models.py         # Định nghĩa 4 kiến trúc CNN & Grad-CAM++
│   └── trainer.py        # Logic huấn luyện mô hình
├── static/               # 🎨 CSS & JS (Frontend)
├── templates/            # 🖼️ Giao diện HTML (Jinja2)
├── checkpoints/          # ⚖️ Lưu trữ trọng số mô hình (.pth)
├── tests/                # 🧪 Kiểm thử tự động
├── app.py                # Flask API & Web Server
├── train.py              # Script chạy huấn luyện
└── requirements.txt      # Danh sách thư viện
```

---

## 🛠️ Cài đặt & Khởi chạy

### 1. Chuẩn bị môi trường

Yêu cầu Python 3.8 trở lên. Khuyến khích sử dụng môi trường ảo (venv):

```bash
# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Khởi động Web Server (Ensemble 4 mô hình)

Sử dụng lệnh sau để chạy server với đầy đủ các checkpoints (điều chỉnh đường dẫn phù hợp):

```powershell
python app.py `
  --resnet checkpoints/resnet50/20260314_130801/resnet50_best.pth `
  --effnet checkpoints/efficientnet/20260314_130441/efficientnet_best.pth `
  --convnext checkpoints/convnext_small/20260320_081724/convnext_small_best.pth `
  --effnet_v2 checkpoints/efficientnet_v2_s/20260320_091400/efficientnet_v2_s_best.pth
```

### 3. Truy cập ứng dụng

Mở trình duyệt tại: **[http://localhost:5000](http://localhost:5000)**

---

## 🏗️ Kiến trúc kỹ thuật

- **Backend**: Flask (Python) phục vụ API và Static files.
- **Deep Learning**: PyTorch framework.
- **Augmentation**: Albumentations (tăng cường dữ liệu MRI).
- **Frontend**: Vanilla JS (ES6+), Modern CSS (Glassmorphism design).
- **Visualization**: Grad-CAM++ (Gradient-weighted Class Activation Mapping).

---

## 🧪 Hệ thống Kiểm thử

Chạy kiểm thử để đảm bảo tính ổn định của các module:

```bash
python -m unittest discover tests
```

---

## 🐳 Triển khai Docker

```bash
# Build image
docker build -t neuroscan-ai .

# Chạy với hỗ trợ GPU
docker run --gpus all -p 5000:5000 neuroscan-ai
```

---

## ⚠️ Khước từ trách nhiệm (Disclaimer)

Sản phẩm này là **công cụ hỗ trợ nghiên cứu** dựa trên trí tuệ nhân tạo. Kết quả phân loại KHÔNG được coi là chẩn đoán y khoa chính thức. Mọi quyết định điều trị cần có sự tham vấn của bác sĩ chuyên khoa.

---

**Hoa Giap - Quoc Dat**
