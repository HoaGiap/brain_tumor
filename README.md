# 🧠 NeuroScan AI — Brain Tumor MRI Classifier

Hệ thống phân loại khối u não trên ảnh MRI sử dụng **ResNet50** và **EfficientNet-B0** với trực quan hóa **Grad-CAM++**. Dự án đã được tối ưu hóa cho việc bảo trì, nâng cấp và triển khai chuyên nghiệp.

---

## 📁 Cấu trúc dự án (Đã Module hóa)

```
brain_tumor/
├── src/
│   ├── config.py         # ⚙️ Cấu hình tập trung (IMG_SIZE, CLASS, Port...)
│   ├── dataset.py        # Data loading, augmentation (Albumentations)
│   ├── models.py         # ResNet50, EfficientNet-B0, Grad-CAM++
│   └── ...
├── static/               # 🎨 Assets cho Frontend
│   ├── css/styles.css    # Định dạng và hiệu ứng giao diện
│   └── js/main.js        # Logic tương tác và gọi API
├── templates/
│   └── index.html        # Giao diện khung (HTML)
├── checkpoints/          # Lưu trữ trọng số mô hình đã huấn luyện
├── tests/                # 🧪 Hệ thống kiểm thử tự động
├── Dockerfile            # 🐳 Cấu hình đóng gói ứng dụng
├── app.py                # Flask API & Web Server
├── train.py              # Script huấn luyện mô hình
└── requirements.txt      # Danh sách thư viện cần thiết
```

---

## ⚙️ Cài đặt & Khởi chạy nhanh

### 1. Cài đặt môi trường

Sử dụng **Python 3.8+** và chạy lệnh sau:

```bash
pip install -r requirements.txt
```

### 2. Chạy Server

Sử dụng lệnh sau để khởi động hệ thống (thay đổi đường dẫn checkpoint của bạn):

```bash
python app.py --resnet checkpoints/resnet50/20260314_130801/resnet50_best.pth --effnet checkpoints/efficientnet/20260314_130441/efficientnet_best.pth
```

### 3. Truy cập giao diện

Mở trình duyệt và truy cập: **[http://localhost:5000](http://localhost:5000)**

> **Lưu ý:** Không mở trực tiếp file `index.html`. Bạn phải truy cập qua URL của server để giao diện hiển thị đúng và kết nối được AI.

---

## 🐳 Triển khai với Docker

Nếu bạn muốn chạy ứng dụng trong môi trường container sạch sẽ:

```bash
# 1. Build image
docker build -t neuroscan-ai .

# 2. Chạy container (Hỗ trợ GPU)
docker run --gpus all -p 5000:5000 neuroscan-ai
```

---

## 🧪 Kiểm thử (Testing)

Đảm bảo logic mô hình và cấu hình luôn ổn định:

```bash
python -m unittest discover tests
```

---

## 🏗️ Kiến trúc & Kỹ thuật

- **Ensemble Learning**: Kết hợp dự đoán từ ResNet50 và EfficientNet-B0 để tăng độ chính xác.
- **Grad-CAM++**: Giải thích quyết định của AI bằng cách quét và hiển thị vùng nghi vấn trên ảnh MRI.
- **Centralized Config**: Mọi tham số quan trọng đều được quản lý tại `src/config.py`.
- **Responsive UI**: Giao diện chế độ tối (Dark mode), hỗ trợ kéo thả ảnh và hiển thị kết quả thời gian thực.

---

## ⚠️ Lưu ý quan trọng

Đây là **công cụ nghiên cứu**. Kết quả chỉ mang tính chất tham khảo và **KHÔNG** thay thế chẩn đoán chuyên môn của bác sĩ.

---

_Phát triển bởi NeuroScan Team · Sử dụng PyTorch, Flask & Albumentations_
