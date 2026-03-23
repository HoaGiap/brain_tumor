# Hướng Dẫn Sử Dụng Docker cho Dự Án NeuroScan AI

Tài liệu này hướng dẫn cách đóng gói và chạy ứng dụng NeuroScan AI (Nhận diện u não qua MRI) bằng Docker. Việc sử dụng Docker giúp đảm bảo môi trường chạy nhất quán, đặc biệt là việc cấu hình GPU (NVIDIA CUDA) cho PyTorch.

---

## 1. Yêu cầu hệ thống

Để chạy dự án này với Docker và hỗ trợ GPU, bạn cần:
1.  **Docker Desktop** (đã cài đặt và bật WSL 2 Backend).
2.  **NVIDIA Container Toolkit** (để Docker có thể truy cập GPU của bạn).
    - Hướng dẫn cài đặt: [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3.  **WSL 2** (Windows Subsystem for Linux 2) đã được cài đặt trên Windows.

---

## 2. Các file cấu hình Docker trong dự án

- `Dockerfile`: Chứa các chỉ dẫn để xây dựng Image (cài đặt Linux, Python, PyTorch, OpenCV, v.v.).
- `docker-compose.yml`: Giúp chạy container dễ dàng hơn với các thiết lập sẵn về cổng (Port), ổ đĩa (Volumes) và GPU.
- `.dockerignore`: Các file không cần thiết (như `venv`, `__pycache__`) sẽ bị bỏ qua khi build image để giảm dung lượng.

---

## 3. Cách chạy bằng Docker Compose (Khuyên dùng)

Đây là cách đơn giản nhất vì mọi cấu hình đã được thiết lập sẵn trong file `docker-compose.yml`.

### Bước 1: Build và Chạy
Mở Terminal (PowerShell hoặc CMD) tại thư mục `brain_tumor` và chạy:
```bash
docker-compose up --build -d
```
- `--build`: Tự động xây dựng lại image nếu có thay đổi trong Dockerfile hoặc source code.
- `-d`: Chạy ngầm (detached mode).

### Bước 2: Kiểm tra trạng thái
```bash
docker ps
```
Bạn sẽ thấy container có tên `brain-tumor-container` đang chạy ở cổng `5000`.

### Bước 3: Truy cập ứng dụng
Mở trình duyệt và truy cập: [http://localhost:5000](http://localhost:5000)

---

## 4. Cách chạy bằng lệnh Docker thủ công

Nếu bạn không muốn dùng Docker Compose:

### Bước 1: Build Image
```bash
docker build -t brain-tumor-app .
```

### Bước 2: Chạy với hỗ trợ GPU
```bash
docker run --gpus all -it -p 5000:5000 \
  -v ${PWD}/checkpoints:/app/checkpoints \
  --name my-brain-app \
  brain-tumor-app \
  python app.py --resnet checkpoints/resnet50/20260314_130801/resnet50_best.pth --effnet checkpoints/efficientnet/20260314_130441/efficientnet_best.pth
```

---

## 5. Các lệnh hữu ích khác

- **Xem log (nhật ký) của server:**
  ```bash
  docker logs -f brain-tumor-container
  ```
- **Dừng và xóa container:**
  ```bash
  docker-compose down
  ```
- **Truy cập vào bên trong container (Terminal của Linux):**
  ```bash
  docker exec -it brain-tumor-container bash
  ```

---

## 6. Lưu ý về Model Checkpoints

Trong file `docker-compose.yml`, thư mục `checkpoints` trên máy của bạn được gắn (mount) vào thư mục `/app/checkpoints` trong container. Nếu bạn huấn luyện mô hình mới và lưu vào thư mục này, container sẽ tự động nhận thấy các file đó mà không cần phải build lại image.
