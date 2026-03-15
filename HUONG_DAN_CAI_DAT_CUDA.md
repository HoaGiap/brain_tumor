# Hướng dẫn Kiểm tra và Cài đặt CUDA cho Huấn luyện Mô hình

Tài liệu này hướng dẫn bạn cách kiểm tra xem máy tính có hỗ trợ CUDA (GPU NVIDIA) không và các bước chi tiết để cài đặt môi trường huấn luyện.

---

## 1. Kiểm tra phần cứng (Hardware Check)

Để sử dụng CUDA, máy bạn **bắt buộc** phải có card đồ họa của **NVIDIA**.

### Cách 1: Dùng Task Manager (Đơn giản nhất)
1. Nhấn `Ctrl + Shift + Esc` để mở Task Manager.
2. Chọn tab **Performance**.
3. Nhìn xuống dưới cùng xem có mục **GPU 0** hoặc **GPU 1** mang tên NVIDIA (ví dụ: RTX 3050, GTX 1650...) hay không.

### Cách 2: Dùng lệnh Command Prompt / PowerShell
Mở terminal và gõ:
```powershell
nvidia-smi
```
- **Nếu hiện bảng thông số**: Máy bạn đã cài Driver và có GPU NVIDIA. Hãy chú ý dòng `CUDA Version: XX.X` ở góc trên bên phải (đây là phiên bản Driver cao nhất mà máy bạn hỗ trợ).
- **Nếu báo lỗi "not recognized"**: Có thể máy không có GPU NVIDIA hoặc chưa cài Driver.

---

## 2. Cài đặt Driver NVIDIA (Nếu chưa có)

Nếu máy có GPU nhưng lệnh `nvidia-smi` không chạy, bạn cần:
1. Truy cập: [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Chọn đúng dòng GPU của máy bạn.
3. Tải về và cài đặt (chọn bản **Game Ready Driver** hoặc **Studio Driver** đều được).

---

## 3. Cài đặt PyTorch hỗ trợ CUDA

Đừng cài PyTorch bằng lệnh `pip install torch` thông thường vì nó sẽ mặc định cài bản CPU. Hãy làm theo các bước sau:

### Bước 1: Gỡ cài đặt bản cũ (Nếu có)
```powershell
pip uninstall torch torchvision torchaudio -y
```

### Bước 2: Truy cập trang chủ PyTorch
Truy cập: [pytorch.org](https://pytorch.org/) và chọn cấu hình:
- **PyTorch Build**: Stable
- **Your OS**: Windows
- **Package**: Pip
- **Language**: Python
- **Compute Platform**: Chọn phiên bản CUDA (thông thường là `CUDA 11.8` hoặc `CUDA 12.1`).

### Bước 3: Chạy lệnh cài đặt
Ví dụ cho CUDA 11.8 (phiên bản ổn định nhất hiện nay):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Kiểm tra xem đã cài đặt thành công chưa

Tạo một file Python nhỏ (ví dụ `check_cuda.py`) hoặc chạy trực tiếp trong terminal:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Current device index: {torch.cuda.current_device()}")
else:
    print("CUDA is NOT available. PyTorch is running on CPU.")
```

---

## 5. Lưu ý quan trọng

- **Phiên bản CUDA**: Bạn không nhất thiết phải cài bộ "CUDA Toolkit" từ website NVIDIA nếu bạn chỉ dùng PyTorch. Thư viện PyTorch cài qua Pip đã bao gồm các file cần thiết để chạy.
- **Worker trên Windows**: Khi dùng CUDA trên Windows, hãy đặt `num_workers=0` trong `DataLoader` của PyTorch để tránh lỗi đa tiến trình (multiprocessing).
- **VRAM**: Nếu gặp lỗi `OutOfMemoryError`, hãy giảm `batch_size` xuống (ví dụ từ 32 xuống 16 hoặc 8).
