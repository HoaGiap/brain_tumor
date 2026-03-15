# Hướng dẫn Khởi chạy Server Flask (Backend)

Tài liệu này hướng dẫn bạn cách chạy server để cung cấp chức năng dự đoán cho ứng dụng Web.

---

## 1. Môi trường yêu cầu
Đảm bảo bạn đã kích hoạt môi trường ảo (nếu có) và đã cài đặt đủ thư viện:
```powershell
pip install -r requirements.txt
```

---

## 2. Lệnh khởi chạy chính thức

Dựa trên các mô hình bạn đã huấn luyện thành công, hãy sử dụng lệnh sau trong Terminal (đứng tại thư mục gốc của dự án):

```powershell
python app.py --resnet checkpoints/resnet50/20260314_130801/resnet50_best.pth --effnet checkpoints/efficientnet/20260314_130441/efficientnet_best.pth
```

### Giải thích các tham số:
- `--resnet`: Đường dẫn tới file trọng số (`.pth`) của mô hình ResNet50.
- `--effnet`: Đường dẫn tới file trọng số (`.pth`) của mô hình EfficientNet-B0.
- `--port`: Cổng chạy server (mặc định là `5000`).
- `--host`: Địa chỉ chạy (mặc định là `0.0.0.0` - có thể truy cập từ máy khác trong cùng mạng).

---

## 3. Cách nhận biết server đã chạy thành công

Khi chạy lệnh trên, bạn sẽ thấy thông báo tương tự như sau trong terminal:

```text
2026-03-14 14:14:19 | INFO | Device: cuda
2026-03-14 14:14:19 | INFO | Loading resnet50 from checkpoints/resnet50/...
2026-03-14 14:14:21 | INFO |   resnet50 loaded ✓
2026-03-14 14:14:21 | INFO | Loading efficientnet from checkpoints/efficientnet/...
2026-03-14 14:14:22 | INFO |   efficientnet loaded ✓
2026-03-14 14:14:22 | INFO | Starting Flask on 0.0.0.0:5000
 * Running on http://127.0.0.1:5000
```

---

## 4. Xử lý lỗi thường gặp

### Lỗi: "Address already in use" (Cổng 5000 đã bị chiếm)
Nếu bạn thấy lỗi báo cổng 5000 đang được sử dụng, bạn có thể đổi cổng khác bằng tham số `--port`:
```powershell
python app.py --resnet ... --effnet ... --port 5001
```
*(Lưu ý: Nếu đổi cổng, bạn cũng cần sửa biến `const API_BASE = "http://localhost:5001"` trong file `templates/index.html`)*.

### Lỗi: "FileNotFoundError"
Hãy kiểm tra kỹ đường dẫn tới file `.pth`. Bạn có thể dùng phím `Tab` để tự động hoàn thành đường dẫn trong terminal để tránh gõ sai.

---

## 5. Sau khi chạy server
Hãy giữ nguyên cửa sổ Terminal này (không được đóng). Sau đó mở file `templates/index.html` bằng trình duyệt để bắt đầu sử dụng.
